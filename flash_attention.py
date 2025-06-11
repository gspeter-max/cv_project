import torch 
import torch.nn as nn 

class flash_attention(nn.Module):
    def __init__(self, dim , query_block_size, key_block_size,group_index_tracking : list = [4, 8, 12, 16]):

        super(flash_attention, self).__init__()
        self.dim = dim
        self.group_index_tracking = group_index_tracking
        self.query_block_size = query_block_size 
        self.key_block_size = key_block_size 

    def forward(self, q, k, v, attn_mask=None):
        # Implement the forward pass for flash attention
        batch_size , seq_len, _ = q.size() 
        query_group = q.view(batch_size * seq_len, self.dim) 
        key_group = k.view(batch_size * seq_len, self.dim) 
        value_group = v.view(batch_size * seq_len, self.dim) 
        final_output = torch.zeros((batch_size * seq_len, self.dim), device=q.device) 
        
        for group_index in range (len(self.group_index_tracking)-1):
            # Extract the current group based on the group index tracking 
            '''
            This is the core of the flash attention mechanism.
            we are flatening the input tensors q, k, v to process them in blocks. 
            and that index tracking is used to define the segments of the input sequence. 
            example:
            If group_index_tracking = [0, 4, 8, 12, 16], then:
                - group 0 corresponds to indices 0 to 4
                - group 1 corresponds to indices 4 to 8
                - group 2 corresponds to indices 8 to 12
                - group 3 corresponds to indices 12 to 16
            
            This allows us to process the input sequence in smaller segments,
            which is essential for efficient attention computation.
            The group_index_tracking is a list that defines the start and end indices of each group.
            It is used to segment the input sequence into manageable blocks for processing.
            The input tensors q, k, v are reshaped to have a shape of (batch_size * seq_len, dim).
            The reshaping is done to flatten the input tensors so that we can process them in blocks.
            We iterate over the groups defined by group_index_tracking.
            Each group corresponds to a segment of the input sequence.
            that we will process in blocks.
            
            '''
            start_group = self.group_index_tracking[group_index] 
            end_group = self.group_index_tracking[group_index + 1]
            query = query_group[start_group:end_group,: ]
            key = key_group[start_group:end_group,: ] 
            value = value_group[start_group:end_group,: ] 
            grouped_batch = end_group - start_group
            
            flash_attention_output = torch.zeros((grouped_batch, self.dim), device=query.device) 

            for i in range(0, grouped_batch, self.query_block_size):
                start_i_idx = i
                end_i_idx = min(i + self.query_block_size, grouped_batch)
                query_block = query[start_i_idx:end_i_idx, :]

                block_size = end_i_idx - start_i_idx
                # preparing Sram
                i_query_output = torch.zeros((block_size, self.dim), device=query.device)
                previous_max = torch.full((block_size, 1), float('-inf'), device=query.device)
                previous_sum = torch.zeros((block_size, 1), device=query.device)

                ''' Iterate over the key blocks
                and we are compute the diff between the previous max and the current max 
                and update the previous sum and previous max accordingly
                so we are change out the point of view according to new max and sum'''
                for j in range(0, grouped_batch, self.key_block_size):
                    # Compute attention for each block
                    start_j_idx = j
                    end_j_idx = min(j + self.key_block_size, grouped_batch)
                    key_block = key[start_j_idx:end_j_idx, :]
                    value_block = value[start_j_idx:end_j_idx, :]

                    score = query_block @ key_block.T / (self.dim ** 0.5)

                    if end_i_idx <= start_j_idx: 
                        """ If the key block starts after the query block ends, skip it
                        This is to ensure that we do not compute attention for keys that are not relevant 
                        to the current query block. """

                        continue
                    if end_i_idx > start_j_idx:

                        """ If the query block and key block overlap, we need to apply a causal mask
                        to ensure that we do not attend to future tokens. """

                        row_index = torch.arange(start_i_idx, end_i_idx).unsqueeze(1)
                        col_index = torch.arange(start_j_idx, end_j_idx).unsqueeze(0) 
                        causal_mask = row_index < col_index 
                        score = score.masked_fill(causal_mask, -float('inf'))

                    __max = torch.max(score, dim=-1, keepdim=True)
                    _new_max = torch.maximum(previous_max, __max.values)

                    _diff_in_max = previous_max - _new_max 
                    _exp_diff_in_max = torch.exp(_diff_in_max)
                    previous_sum = previous_sum * _exp_diff_in_max
                    i_query_output = i_query_output * _exp_diff_in_max
                    attention_weight = torch.exp(score - _new_max)
                    previous_max = _new_max
                    previous_sum = previous_sum + attention_weight.sum(dim=-1, keepdim=True)

                    i_query_output += attention_weight @ value_block 
                '''
                this is becuase the previous_sum is the sum of the attention weights
                becuase that is same for all the query blocks
                so we are take the common and do that in at the end '''
                
                i_query_output = i_query_output / previous_sum
                flash_attention_output[start_i_idx:end_i_idx, :] = i_query_output

            final_output[start_group:end_group, :] = flash_attention_output 
        final_output = final_output.view(batch_size, seq_len, self.dim) 
        # Create this INSIDE your forward method for a correct comparison
        with torch.no_grad():
    # This will be our real target
            ground_truth_attention = torch.zeros_like(q)

            for group_index in range(len(self.group_index_tracking) - 1):
                start_group = self.group_index_tracking[group_index]
                end_group = self.group_index_tracking[group_index + 1]

                # Get the original Q, K, V for this specific group
                q_group_real = q.view(-1, self.dim)[start_group:end_group, :]
                k_group_real = k.view(-1, self.dim)[start_group:end_group, :]
                v_group_real = v.view(-1, self.dim)[start_group:end_group, :]

                group_len = q_group_real.shape[0]

                # --- Standard, Inefficient, but CORRECT Causal Attention for this Group ---
                scores = q_group_real @ k_group_real.T / (self.dim ** 0.5)
                
                # Create a proper causal mask for this group's size
                causal_mask = torch.triu(torch.ones(group_len, group_len, device=q.device), diagonal=1).bool()
                scores.masked_fill_(causal_mask, -float('inf'))
                
                attn_weights = torch.softmax(scores, dim=-1)
                
                output_group = attn_weights @ v_group_real
                # Place the result into our ground truth tensor
                ground_truth_attention.view(-1, self.dim)[start_group:end_group, :] = output_group
            ground_truth_attention = ground_truth_attention.view(batch_size, seq_len, self.dim)

                # NOW, compare against this new target
        print(f'how much close : {torch.allclose(final_output, ground_truth_attention, atol=1e-5)}')
        print(f'max diff : {torch.max(torch.abs(final_output - ground_truth_attention))}')

        return final_output

# Example usage
if __name__ == "__main__":
    batch_size = 30
    seq_len = 100
    dim = 5120
    query_block_size = 50
    key_block_size = 50

    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)

    model = flash_attention(dim, query_block_size, key_block_size)
    output = model(query, key, value)
    print(output.shape)  # Should be (batch_size, seq_len, dim) 
