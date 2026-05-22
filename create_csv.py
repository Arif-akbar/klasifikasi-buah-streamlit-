import os
import pandas as pd

def generate_fruit_csv(root_path, output_name):
    data = []
    # Mapping label string ke angka untuk model
    class_map = {
        'freshapples': 0, 'freshbanana': 1, 'freshoranges': 2,
        'rottenapples': 3, 'rottenbanana': 4, 'rottenoranges': 5
    }
    
    for label_name, label_id in class_map.items():
        folder_path = os.path.join(root_path, label_name)
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'file': f"{root_path}/{label_name}/{img_name}",
                        'label': label_id,
                        'category': label_name
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_name, index=False)
    print(f"✅ Berhasil membuat {output_name} dengan {len(df)} baris.")

if __name__ == "__main__":
    generate_fruit_csv('dataset/train', 'fruit_train.csv')
    generate_fruit_csv('dataset/test', 'fruit_test.csv')