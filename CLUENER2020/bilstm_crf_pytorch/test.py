# batch = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
# print(batch)
# # print('zip(*batch)', zip(*batch))
# print('list(zip(*batch))', list(zip(*batch)))

# ****************************************************

# import numpy as np
# attempt = [([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5'), 
#            ([1], [2], 3, '4', '5')]
# print(np.shape(attempt))

# ****************************************************

eval_info = {'acc': 2, 'recall': 3, 'f1': 4}
eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
result = {'eval_loss': 1}
result = dict(result, **eval_info)
print(result)
