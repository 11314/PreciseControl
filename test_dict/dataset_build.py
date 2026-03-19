import os
import shutil
from pathlib import Path

def build_training_dataset():
    """
    构建训练数据集的主函数
    """
    # 定义所有路径
    source_aligned_path = "/hxp/lxy/dataset/aligned_faces"  # 源数据集原图路径
    edited_images_path = "/hxp/lxy/dataset/CelebA_EditedImages"  # 属性编辑后数据集路径
    target_dataset_path = "/hxp/zy/PreciseControl/test_dict/dataset"  # 目标数据集路径
    
    # 指定的图片ID列表（共29个）
    selected_ids = [
        '32', '37', '40', '43', '52', '62', '89', '92', '94', '97', 
        '113', '138', '142', '162', '167', '168', '178', '180', 
        '194', '200', '240', '243', '244', '247', '249', '256', '264', '272'
    ]
    
    # 确保目标目录存在
    os.makedirs(target_dataset_path, exist_ok=True)
    
    print(f"开始构建数据集...")
    print(f"源原图路径: {source_aligned_path}")
    print(f"源编辑图路径: {edited_images_path}")
    print(f"目标路径: {target_dataset_path}")
    print(f"选择的ID数量: {len(selected_ids)}")
    
    # 创建日志记录成功和失败的文件
    success_log = []
    error_log = []
    
    for idx, img_id in enumerate(selected_ids):
        print(f"\n处理第 {idx+1}/{len(selected_ids)} 个图片: {img_id}")
        
        try:
            # 步骤1: 处理原图
            original_img_source = os.path.join(source_aligned_path, f"{img_id}.jpg")
            
            # 检查原图是否存在
            if not os.path.exists(original_img_source):
                error_msg = f"原图不存在: {original_img_source}"
                print(f"错误: {error_msg}")
                error_log.append(error_msg)
                continue
            
            # 在目标数据集中创建以ID命名的文件夹
            person_folder = os.path.join(target_dataset_path, img_id)
            os.makedirs(person_folder, exist_ok=True)
            
            # 将原图复制到对应文件夹并重命名为 "original.jpg"
            original_img_target = os.path.join(person_folder, "original.jpg")
            shutil.copy2(original_img_source, original_img_target)
            print(f"✓ 原图已复制: {original_img_source} -> {original_img_target}")
            
            # 步骤2: 处理属性编辑后的图片
            edited_folder_source = os.path.join(edited_images_path, img_id)
            
            # 检查编辑图片文件夹是否存在
            if not os.path.exists(edited_folder_source):
                error_msg = f"编辑图片文件夹不存在: {edited_folder_source}"
                print(f"错误: {error_msg}")
                error_log.append(error_msg)
                continue
            
            # 在编辑文件夹中查找 "face with beard.jpg" 文件
            beard_img_source = os.path.join(edited_folder_source, "face with beard.jpg")
            
            if not os.path.exists(beard_img_source):
                error_msg = f"beard图片不存在: {beard_img_source}"
                print(f"错误: {error_msg}")
                error_log.append(error_msg)
                continue
            
            # 将beard图片复制到目标文件夹并重命名为 "ID_beard.jpg"
            beard_img_target = os.path.join(person_folder, f"{img_id}_beard.jpg")
            shutil.copy2(beard_img_source, beard_img_target)
            print(f"✓ Beard图片已复制: {beard_img_source} -> {beard_img_target}")
            
            success_log.append(img_id)
            print(f"✓ ID {img_id} 处理完成")
            
        except Exception as e:
            error_msg = f"处理ID {img_id} 时出错: {str(e)}"
            print(f"错误: {error_msg}")
            error_log.append(error_msg)
    
    # 输出总结报告
    print(f"\n{'='*50}")
    print("数据集构建完成!")
    print(f"成功处理的ID数量: {len(success_log)}")
    print(f"失败的ID数量: {len(error_log)}")
    
    if success_log:
        print(f"成功的ID: {', '.join(success_log)}")
    
    if error_log:
        print(f"错误日志:")
        for error in error_log:
            print(f"  - {error}")
    
    # 验证最终结果
    verify_dataset_structure(target_dataset_path, selected_ids)

def verify_dataset_structure(target_path, expected_ids):
    """
    验证目标数据集结构是否正确
    """
    print(f"\n{'='*30}")
    print("验证数据集结构...")
    
    missing_folders = []
    incomplete_folders = []
    
    for img_id in expected_ids:
        folder_path = os.path.join(target_path, img_id)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            missing_folders.append(img_id)
            continue
        
        # 检查文件夹中的文件
        files_in_folder = os.listdir(folder_path)
        required_files = {"original.jpg", f"{img_id}_beard.jpg"}
        existing_files = set(files_in_folder)
        
        if not required_files.issubset(existing_files):
            incomplete_folders.append({
                'id': img_id,
                'missing': required_files - existing_files,
                'existing': existing_files
            })
    
    print(f"预期的ID数量: {len(expected_ids)}")
    print(f"存在的文件夹数量: {len([f for f in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, f))])}")
    
    if missing_folders:
        print(f"缺少的文件夹: {missing_folders}")
    
    if incomplete_folders:
        print("不完整的文件夹:")
        for folder_info in incomplete_folders:
            print(f"  - ID {folder_info['id']}: 缺少 {folder_info['missing']}, 存在 {folder_info['existing']}")
    else:
        print("✓ 所有文件夹结构完整!")

def print_sample_structure(target_path):
    """
    打印样本结构供参考
    """
    print(f"\n{'='*30}")
    print("目标数据集结构示例:")
    print(f"""
{target_path}/
├── 32/
│   ├── original.jpg          # 来自 /hxp/lxy/dataset/aligned_faces/32.jpg
│   └── 32_beard.jpg          # 来自 /hxp/lxy/dataset/CelebA_EditedImages/32/face with beard.jpg
├── 37/
│   ├── original.jpg          # 来自 /hxp/lxy/dataset/aligned_faces/37.jpg
│   └── 37_beard.jpg          # 来自 /hxp/lxy/dataset/CelebA_EditedImages/37/face with beard.jpg
├── 40/
│   ├── original.jpg
│   └── 40_beard.jpg
...
└── 272/
    ├── original.jpg
    └── 272_beard.jpg
    """)

if __name__ == "__main__":
    # 执行数据集构建
    build_training_dataset()
    
    # 打印结构示例
    target_dataset_path = "/hxp/zy/PreciseControl/test_dict/dataset"
    print_sample_structure(target_dataset_path)
