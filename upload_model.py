from huggingface_hub import HfApi
import os


base_repo_id = "models/checkpoint-544"
upload_name = "amuvarma/convo-tts-tune-76layer"


def push_folder_to_hub(local_folder, repo_id, commit_message="Update model"):
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return None

    try:
        uploaded_files = []
        for root, _, files in os.walk(local_folder):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_folder)
                print(f"Uploading {rel_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    commit_message=commit_message
                )
                uploaded_files.append(rel_path)

        
        print(f"Successfully uploaded {len(uploaded_files)} files to {repo_id}")
        return api.get_full_repo_name(repo_id)
    except Exception as e:
        print(f"Error during upload: {e}")
        return None


push_folder_to_hub(f"./{base_repo_id}", upload_name, "Update model")
