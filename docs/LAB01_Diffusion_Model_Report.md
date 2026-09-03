# BÁO CÁO LAB01: CÀI ĐẶT DIFFUSION MODEL VÀ SINH ẢNH

**Môn học:** Làm quen với AI Engineer  
**Nhóm:** `10`  
**Thành viên:** `Phan Ngọc Đức Huy`, `Dũng`, `Đức`  

## 1. Mục tiêu

- Kiểm tra khả năng chạy mô hình AI bằng GPU NVIDIA trên máy Windows.
- Tạo môi trường Python độc lập cho bài lab.
- Cài PyTorch có CUDA và các thư viện `diffusers`, `transformers`, `accelerate`, `safetensors`.
- Tải Stable Diffusion v1.5 từ Hugging Face.
- Sinh và lưu ảnh từ prompt:

> Spider-Man sitting on a skyscraper edge, back to camera, New York skyline at sunset, cinematic wide shot


## 2. Kiểm tra máy trước khi cài đặt

Các lệnh kiểm tra ban đầu:

```powershell
python --version
py --version
nvidia-smi
nvcc --version
```

Kết quả thực tế:

| Thành phần | Trạng thái ban đầu |
|---|---|
| Python | Đã cài Python 3.13.15; kiểm tra thành công bằng `py --version` trong Command Prompt |
| GPU | NVIDIA GeForce GTX 1650 Ti with Max-Q Design |
| VRAM | 4096 MiB (4 GB) |
| NVIDIA Driver | 595.79 |
| CUDA compatibility do driver báo | 13.2 |
| CUDA Toolkit / `nvcc` | Chưa cài |

Con số “CUDA Version 13.2” trong `nvidia-smi` là mức CUDA cao nhất mà driver hỗ trợ, không chứng minh CUDA Toolkit đã được cài. Bài lab dùng PyTorch wheel dựng sẵn cho CUDA 12.8; wheel này đã chứa CUDA runtime cần để inference, vì vậy không cần `nvcc` nếu không biên dịch PyTorch hay CUDA extension từ mã nguồn.

## 3. Tạo môi trường Python ảo

Máy đã có Python 3.13.15 và có thể gọi qua Python Launcher (`py`). Môi trường `.venv` của dự án được tạo trực tiếp từ bản Python global. File `.venv/pyvenv.cfg` xác nhận đường dẫn Python gốc là:

```text
C:\Users\huyph\AppData\Local\Programs\Python\Python313\python.exe
```

Trên Command Prompt của máy, có thể tạo môi trường mới bằng:

```powershell
cd D:\Workspace\Projects\diffusion-model-lab01
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip --version
```

Nếu làm bài trên một máy mới chưa có Python, tải bản x64 từ [python.org](https://www.python.org/downloads/windows/) rồi chạy quy trình tương tự. Lệnh tổng quát là:

```powershell
cd D:\Workspace\Projects\diffusion-model-lab01
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip --version
```

Kết quả trong `.venv`:

```text
Python 3.13.15
pip 26.2.1
```

Nếu PowerShell không cho phép chạy `Activate.ps1`, có thể gọi trực tiếp Python trong môi trường ảo mà không cần kích hoạt:

```powershell
.\.venv\Scripts\python.exe --version
```

## 4. Cài thư viện

File `requirements.txt` đã khóa các phiên bản được kiểm thử:

```text
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.11.0+cu128
torchvision==0.26.0+cu128
diffusers==0.40.0
transformers==5.16.1
accelerate==1.14.0
safetensors==0.8.0
```

Cài toàn bộ thư viện:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Kết quả:

```text
No broken requirements found.
2.11.0+cu128
12.8
True
NVIDIA GeForce GTX 1650 Ti with Max-Q Design
```

Như vậy PyTorch đã truy cập GPU thành công. Phiên bản CUDA của wheel PyTorch là 12.8 và tương thích với driver hiện tại.

## 5. Tải mô hình Stable Diffusion v1.5

Đoạn mẫu của đề dùng model ID cũ:

```python
"runwayml/stable-diffusion-v1-5"
```

Kho RunwayML cũ đã bị ngừng duy trì. Bài lab dùng model ID hiện tại trên Hugging Face:

```python
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
```

Lần chạy đầu, `StableDiffusionPipeline.from_pretrained(...)` tự tải 15 tệp của model. Cache được đặt trong dự án tại `.cache/huggingface/hub`, có tổng dung lượng khoảng **5,11 GiB**. Các lần chạy sau dùng lại cache, không tải lại trọng số.

## 6. Chương trình sinh ảnh

Mã đầy đủ nằm trong file `generate_image.py`. Phần chính của chương trình:

```python
    gpu_name = torch.cuda.get_device_name(0)
    if args.precision == "auto":
        dtype = torch.float32 if "GTX 16" in gpu_name.upper() else torch.float16
    else:
        dtype = getattr(torch, args.precision)
    print(f"Precision: {dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        dtype=dtype,
        use_safetensors=True,
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    extrema = image.convert("RGB").getextrema()
    if all(channel_max <= 1 for _, channel_max in extrema):
        raise RuntimeError(
            "The generated image is black. Retry with --precision float32 and review "
            "any safety-checker message printed above."
        )

    output_path = args.output or next_output_path(Path(__file__).resolve().parent / "outputs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image: {output_path.resolve()}")
```

Các lựa chọn cấu hình:

- `768 × 512`: khung ngang phù hợp “cinematic wide shot”; cả hai cạnh đều là bội số của 8.
- `25` inference steps: cân bằng chất lượng và thời gian trên GPU phổ thông.
- `guidance_scale=7.5`: giúp ảnh bám prompt ở mức hợp lý.
- `seed=42`: kết quả có thể tái lập khi phần mềm và phần cứng không đổi.
- `use_safetensors=True`: ưu tiên định dạng trọng số an toàn hơn pickle.
- `enable_sequential_cpu_offload()`: chuyển từng lớp giữa RAM và GPU để giảm đỉnh VRAM, phù hợp GPU 4 GB.
- Safety checker của pipeline được giữ nguyên và đã chạy thành công ở lần sinh cuối.
- `next_output_path(...)`: tự chọn `output1.png`, `output2.png`, ... để không ghi đè ảnh từ lần chạy trước.

## 7. Chạy sinh ảnh

Từ thư mục gốc của dự án:

```powershell
.\.venv\Scripts\Activate.ps1
python generate_image.py
```

Hoặc không kích hoạt môi trường:

```powershell
.\.venv\Scripts\python.exe .\generate_image.py
```

Chương trình cũng cho phép thay prompt và các tham số:

```powershell
python generate_image.py --prompt "A futuristic city at night" --seed 123 --steps 30
```

## 8. Kết quả thực nghiệm

Lần chạy cuối bằng Python 3.13.15 hoàn tất đủ `25/25` bước trong chế độ offline, phần khuếch tán mất khoảng **1 phút 13 giây**. File đầu ra:

- Đường dẫn: `outputs/spiderman_sunset.png`
- Định dạng: PNG
- Kích thước: 768 × 512 pixel
- Chế độ màu: RGB
- Dung lượng: 818.907 byte
- SHA-256: `b899d22d73d16d9b68bea0509e3fe38a777eee232a631d5febe84cdb813507ad`

![Ảnh Spider-Man được sinh bằng Stable Diffusion v1.5](../outputs/spiderman_sunset.png)

Ảnh thể hiện Spider-Man ngồi trên mép tòa nhà, phía sau là toàn cảnh các tòa nhà cao tầng ở New York. Stable Diffusion v1.5 đã thể hiện đúng chủ thể và góc máy rộng, nhưng tư thế “back to camera” và ánh sáng hoàng hôn chưa hoàn toàn chính xác. Đây là giới hạn thường gặp của model cũ khi prompt chứa nhiều ràng buộc bố cục.

## 9. Lỗi gặp phải và cách xử lý

### 9.1. API VAE slicing thay đổi

Lệnh `pipe.enable_vae_slicing()` không tồn tại trong Diffusers 0.40.0 và gây `AttributeError`. Cách gọi đúng với phiên bản đã cài:

```python
pipe.vae.enable_slicing()
```

### 9.2. Ảnh đen khi dùng FP16

Lần thử đầu với `torch.float16` xuất hiện cảnh báo `invalid value encountered in cast` và ảnh toàn màu đen. Việc tắt safety checker không giải quyết được lỗi, cho thấy nguyên nhân là giá trị NaN trong pipeline FP16 chứ không phải nội dung prompt.

Trên GTX 1650 Ti, chuyển sang `torch.float32` khắc phục lỗi. GPU này không có Tensor Cores nên FP32 trong phép thử còn nhanh hơn FP16. Script tự chọn FP32 nếu tên GPU chứa `GTX 16`; người dùng vẫn có thể ép kiểu bằng `--precision float16` hoặc `--precision float32`.

### 9.3. VRAM chỉ có 4 GB

Đưa toàn bộ pipeline lên GPU bằng `.to("cuda")` có nguy cơ hết VRAM, đặc biệt khi Windows đang dùng GPU để hiển thị. Vì vậy chương trình dùng sequential CPU offload. Cách này cần RAM hệ thống và truyền dữ liệu CPU–GPU nhiều hơn, nhưng đáng tin cậy trên cấu hình hiện tại.

### 9.4. Tải model trên Windows

Hugging Face cảnh báo cache không dùng được symbolic link nếu Windows Developer Mode chưa bật. Đây không phải lỗi; model vẫn tải và chạy được nhưng cache có thể tốn thêm dung lượng.

## 10. Kết luận

Môi trường Python độc lập dựa trên Python global 3.13.15 đã được tạo, các thư viện không có dependency bị hỏng, PyTorch nhận GPU CUDA thành công và Stable Diffusion v1.5 đã sinh được ảnh PNG từ prompt yêu cầu. Cấu hình GTX 1650 Ti 4 GB chạy được model nhờ sequential CPU offload và FP32; không cần cài riêng CUDA Toolkit cho tác vụ inference với PyTorch wheel dựng sẵn. Model được lưu trong cache cục bộ của dự án, không phụ thuộc cache của Codex.

## 11. Tài liệu tham khảo

1. [Hugging Face Diffusers trên GitHub](https://github.com/huggingface/diffusers)
2. [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
3. [PyTorch – Start Locally](https://pytorch.org/get-started/locally/)
4. [Diffusers – Reduce memory usage](https://huggingface.co/docs/diffusers/optimization/memory)
5. [Python releases for Windows](https://www.python.org/downloads/windows/)
