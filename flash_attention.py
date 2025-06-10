import torch   
import torch.nn as nn 

class flash_attention(nn.Module): 

    def __init__(self, query_block_size, key_block_size) :
        super().__init__() 
        self.query_dim = query_block_size 
        self.key_dim =  key_block_size 

    def forward(self, query, key, value): 
        # Ensure query, key, value are 3D tensors
        output_matrix = torch.zeros(query.shape) 
        num_rows , head_dims = query.shape 

        for i in range(0, num_rows,self.query_dim): 
            start_row = i 
            end_row = min(i + self.query_dim, num_rows) 

            query_block  = query[start_row:end_row, :] 
            previous_sum = torch.zeros((query_block.shape[0],))
            previous_max = torch.full((query_block.shape[0],), float('-inf'))
            output_block = torch.zeros_like(query_block) 

            for j in range(0, num_rows, self.key_dim): 
                start_key = j 
                end_key = min(j + self.key_dim, num_rows)  
                key_block = key[start_key:end_key,:] 
                value_block = value[start_key:end_key,:]

                score_value = (query_block @ key_block.T) / torch.sqrt(torch.tensor(head_dims, dtype=query.dtype))
                max_value = torch.max(score_value, dim=1).values
                new_max = torch.maximum(previous_max, max_value)
                temp = torch.exp(previous_max - new_max)   
                
                previous_sum = previous_sum * temp 
                output_block = output_block * temp.unsqueeze(1) 

                attention_score = torch.exp(score_value - new_max.unsqueeze(1))
                previous_sum += torch.sum(attention_score, dim=1)

                previous_max = new_max
                output_block += attention_score @ value_block

            output_matrix[start_row:end_row, :] = output_block / (previous_sum.unsqueeze(1) + 1e-8) 

        attention_output = (query @ key.T) / torch.sqrt(torch.tensor(head_dims, dtype=query.dtype))
        attention_output = torch.softmax(attention_output, dim=-1) @ value

        print("we are close")
        print(torch.allclose(output_matrix, attention_output, atol=1e-6)) 
        print("maximum diff")
        print(torch.max(torch.abs(output_matrix - attention_output))) 
        print(f'attention_output shape: {attention_output}'); print(f' output_matrix shape: {output_matrix}')
        return attention_output == output_matrix


obj = flash_attention(32, 32)  # Example dimensions for query and key blocks
query = torch.randn(64, 64)  # Example query tensor
key = torch.randn(64, 64)     # Example key tensor
value = torch.randn(64, 64)   # Example value tensor
print(obj.forward(query, key, value))
