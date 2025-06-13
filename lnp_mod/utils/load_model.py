import os
import gdown

def load_model_files(model_dir, model_config):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_path = os.path.join(model_dir, model_config['filename'])
    if not os.path.exists(file_path):
        gdown.download(model_config['url'], file_path, quiet=False)

    # check file is downloaded
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found. Download failed.")

    return file_path
    
