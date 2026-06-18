import os
import shutil

# --- KONFIGURASI FOLDER ---
# Ganti teks di bawah ini dengan lokasi folder tempat kamu mengekstrak FGNET dari Kaggle tadi
# Contoh: "C:/Users/Dean/Downloads/FGNET/images"
SUMBER_FGNET = "C:/Users/Dean/Downloads/archive (1)/FGNET/images" 

# Folder tujuan (otomatis masuk ke folder dataset di dalam facedetector)
TUJUAN_DATASET = "dataset" 

def rapikan_dataset():
    print("Mulai merapikan dataset FGNET...")
    
    # Buat folder dataset utama jika belum ada
    if not os.path.exists(TUJUAN_DATASET):
        os.makedirs(TUJUAN_DATASET)
        print(f"📁 Membuat folder utama: {TUJUAN_DATASET}")

    # Cek apakah folder sumber valid
    if not os.path.exists(SUMBER_FGNET):
        print(f"❌ ERROR: Folder sumber '{SUMBER_FGNET}' tidak ditemukan!")
        print("Silakan edit file ini dan pastikan lokasi foldernya sudah benar.")
        return

    jumlah_file = 0

    # Mulai membaca isi folder FGNET yang berantakan
    for filename in os.listdir(SUMBER_FGNET):
        if filename.upper().endswith((".JPG", ".JPEG", ".PNG")):
            # Mengambil 3 karakter pertama sebagai ID Orang (contoh: dari '001A14.JPG' jadi '001')
            person_id = filename[:3]
            
            if not person_id.isdigit():
                continue
                
            nama_folder_orang = f"orang_{person_id}"
            path_folder_orang = os.path.join(TUJUAN_DATASET, nama_folder_orang)
            
            if not os.path.exists(path_folder_orang):
                os.makedirs(path_folder_orang)
                
            file_sumber = os.path.join(SUMBER_FGNET, filename)
            file_tujuan = os.path.join(path_folder_orang, filename)
            
            shutil.copy(file_sumber, file_tujuan)
            jumlah_file += 1
            
            print(f"✔️ Memindahkan {filename} -> {nama_folder_orang}/")

    print(f"\n✅ SELESAI! Berhasil mengelompokkan {jumlah_file} foto ke dalam folder dataset.")

if __name__ == "__main__":
    rapikan_dataset()