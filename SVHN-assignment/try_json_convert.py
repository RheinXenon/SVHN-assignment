import json
import h5py

def convert_mat_to_json(mat_file, output_json):
    with h5py.File(mat_file, 'r') as f:
        structs = f['digitStruct']['bbox']
        names = f['digitStruct']['name']
        
        def get_name(i):
            name_ref = names[i][0]
            return ''.join(chr(c[0]) for c in f[name_ref])
        
        def get_bbox(i):
            attr = {}
            bbox = f[structs[i][0]]
            for key in ['height', 'label', 'left', 'top', 'width']:
                attr[key] = [f[bbox[key][j][0]][0][0] for j in range(len(bbox[key]))] if len(bbox[key]) > 1 else [bbox[key][0][0]]
            return attr
        
        data = []
        for i in range(len(names)):
            data.append({
                'filename': get_name(i),
                'bbox': get_bbox(i)
            })
    
    # 将数据保存为 JSON 文件
    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Saved to {output_json}")

# 示例：将 train_digitStruct.mat 转换为 JSON 文件
# mat_file = './datasets/train_digitStruct.mat'
# output_json = './datasets/train_digitStruct.json'
mat_file = './datasets/extra_digitStruct.mat'
output_json = './datasets/extra_digitStruct.json'
convert_mat_to_json(mat_file, output_json)
