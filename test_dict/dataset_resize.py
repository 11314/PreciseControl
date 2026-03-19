import os
from PIL import Image
import glob

def resize_beard_images_to_512(base_path):
    """
    将数据集中所有的 beard 图片调整为 512x512
    """
    print(f"开始处理 beard 图片尺寸调整...")
    print(f"基础路径: {base_path}")
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, f))]
    
    processed_count = 0
    error_count = 0
    
    for folder in subfolders:
        folder_path = os.path.join(base_path, folder)
        
        # 查找编号_beard.jpg格式的文件
        beard_files = []
        for file in os.listdir(folder_path):
            if file.endswith('_beard.jpg') and file.replace('_beard.jpg', '') == folder:
                beard_files.append(file)
        
        if not beard_files:
            print(f"警告: 文件夹 {folder} 中未找到匹配的 beard 图片")
            continue
        
        for beard_file in beard_files:
            beard_path = os.path.join(folder_path, beard_file)
            
            try:
                # 打开图片
                with Image.open(beard_path) as img:
                    original_size = img.size
                    print(f"处理: {beard_path}, 原始尺寸: {original_size}")
                    
                    # 检查当前尺寸，如果是1024则调整为512
                    if original_size[0] == 1024 and original_size[1] == 1024:
                        # 调整为512x512
                        resized_img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        
                        # 保存调整后的图片（覆盖原文件）
                        resized_img.save(beard_path, quality=95, optimize=True)
                        
                        # 验证调整结果
                        with Image.open(beard_path) as verify_img:
                            new_size = verify_img.size
                            print(f"✓ 调整完成: {original_size} -> {new_size}")
                        
                        processed_count += 1
                    else:
                        print(f"- 尺寸已正确或不需要调整: {original_size}")
                        
            except Exception as e:
                print(f"✗ 处理 {beard_path} 时出错: {str(e)}")
                error_count += 1
    
    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"成功调整的图片数量: {processed_count}")
    print(f"处理错误的图片数量: {error_count}")

def verify_image_sizes(base_path):
    """
    验证所有图片的尺寸
    """
    print(f"\n验证图片尺寸...")
    
    subfolders = [f for f in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, f))]
    
    for folder in subfolders:
        folder_path = os.path.join(base_path, folder)
        
        # 检查 original.jpg 和 _beard.jpg 的尺寸
        files_to_check = []
        for file in os.listdir(folder_path):
            if file == 'original.jpg' or file.endswith('_beard.jpg'):
                files_to_check.append(file)
        
        for file in files_to_check:
            file_path = os.path.join(folder_path, file)
            try:
                with Image.open(file_path) as img:
                    size = img.size
                    status = "✓" if size[0] == 512 and size[1] == 512 else "✗"
                    print(f"{status} {folder}/{file}: {size}")
            except Exception as e:
                print(f"✗ 无法读取 {file_path}: {str(e)}")

def batch_resize_with_backup(base_path, backup=True):
    """
    批量调整尺寸，可选择备份原图
    """
    if backup:
        backup_path = base_path + "_backup"
        print(f"创建备份到: {backup_path}")
        import shutil
        shutil.copytree(base_path, backup_path)
        print("备份完成!")
    
    resize_beard_images_to_512(base_path)
    verify_image_sizes(base_path)

if __name__ == "__main__":
    # 数据集路径
    dataset_path = "/hxp/zy/PreciseControl/test_dict/dataset"
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 路径不存在: {dataset_path}")
        exit(1)
    
    print("开始调整 beard 图片尺寸...")
    
    # 方案1: 直接调整（推荐）
    resize_beard_images_to_512(dataset_path)
    
    # 方案2: 带备份的调整（更安全）
    # batch_resize_with_backup(dataset_path, backup=True)
    
    # 验证结果
    verify_image_sizes(dataset_path)
    
    print(f"\n{'='*50}")
    print("尺寸调整任务完成!")
