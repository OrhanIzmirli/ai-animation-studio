# MILO teaser — üretim sisteminin tam anlatımı (2026-10-06)

Bu belge, `milo_teaser_hd.mp4` (25 sn, 1920×1056, 24 fps, sesli) filmini
üreten sistemi uçtan uca anlatır: hangi model ne yapıyor, hangi script hangi
sırayla çalışıyor, hangi kararlar neden alındı, sınırlar neler.

---

## 1. Özet: mimari

Yapay zekâ yalnızca **resim** üretir. **Hareketi** hiçbir video modeli
uydurmaz; tamamı kodla tasarlanır (1980'ler cel animasyonu mantığı: boyalı
sabit arka plan + üstünde ayrı katmanda karakter).

```
[SDXL + 80s LoRA]  ──► karakter tasarımı (Milo, tek çizim, önden)
        │                      │
        │                      ├─ gözler elle (kodla) turkuaza boyanır
        │                      └─ arka planından kesilir (düz gri zemin)
        │
        └─► karaktersiz boyalı arka planlar (orman, kovuk, açıklık, gökyüzü, şato)
                               │
[Real-ESRGAN anime] ──► hepsi 4× büyütülür → 1920×1056 / 2048×2048
[Depth-Anything-V2] ──► her arka plan için derinlik haritası
                               │
[ai_studio.puppet]  ──► Milo "kukla": gövde + 2 kol katmanı, kodla çizilen yüz,
                        kumaş gibi bükülme, yöne göre esneme
                               │
[make_milo_teaser.py]  ──► oyunculuk/zamanlama/kamera (dünya koordinatlarında)
[make_milo_teaser_hd.py] ─► HD render: 2.5D paralaks, cel gölge, hareket
                        bulanıklığı, atmosfer, bloom, film greni → teaser.mp4
                               │
[Kokoro-82M TTS] + numpy sentez (efekt, müzik, ortam) ──► miks → teaser_with_audio.mp4
```

Video modeli (Wan 2.2) bu sürümde **kullanılmıyor**. Önceki denemelerde
(milo_01, milo_02) Wan ile üretilen kliplerde karakter her sahnede değişti,
bacak/el çıktı, renk kaydı, hareket rastgeleydi; bu yüzden terk edildi.

---

## 2. Donanım ve kurulu araçlar

| | |
|---|---|
| Makine | RTX 4060 Laptop 8 GB VRAM, 15.7 GB RAM, Windows 11 |
| Proje ortamı | Python 3.11.9, `.venv` (numpy, Pillow, PyYAML, pytest); ffmpeg/ffprobe 8.1.2 |
| ComfyUI | v0.38.2, `~/ai-tools/ComfyUI` (kendi venv'i, torch cu128). HTTP API ile sürülür, arayüz kullanılmaz |
| Kokoro ortamı | `~/ai-tools/kokoro-venv` (yolu `KOKORO_PYTHON` ile değiştirilebilir) (Python 3.11, torch 2.14 CPU, kokoro 0.9.4, transformers 5.19). TTS ve derinlik tahmini burada çalışır |

## 3. Modeller

| Model | Dosya / kaynak | Ne için | Lisans / not |
|---|---|---|---|
| SDXL base 1.0 | `models/checkpoints/sd_xl_base_1.0.safetensors` (sha256 `31e35c80…`) | Tüm resimler | CreativeML OpenRAIL++ |
| "80s cartoon SDXL 1.0" LoRA | Civitai model 125857, `models/loras/civitai_80scartoon_sdxl.safetensors` (sha256 `6a7d2e24…`), güç 1.0, tetik `style of 80scartoon` | 80'ler cel + boyalı arka plan görünümü | Civitai lisans koşulları |
| IP-Adapter plus SDXL ViT-H + CLIP ViT-H | `models/ipadapter/…`, `models/clip_vision/…` | Milo'nun kimliğini sahneler arası tutmak (keyframe denemelerinde) | Final filmde Milo zaten tek çizimden türetildiği için gerekmedi |
| Real-ESRGAN x4plus anime 6B | `models/upscale_models/RealESRGAN_x4plus_anime_6B.pth` (sha256 `f872d837…`) | Arka planları ve Milo'yu 4× büyütüp keskinleştirme | BSD-3 |
| Depth-Anything-V2-Small (HF) | `depth-anything/Depth-Anything-V2-Small-hf`, kokoro-venv içinde transformers ile | Arka planlara derinlik haritası (2.5D kamera) | Apache-2.0 |
| Kokoro-82M | `hexgrad/Kokoro-82M`, ses `am_michael`, hız 0.92 | İngilizce anlatıcı sesi | Apache-2.0; denenen 3 sesten en kalını (F0 ≈ 123 Hz) seçildi |

## 4. Kilitli stil (style lock)

`configs/style_locks/milo_80s_v1.yaml` → v2 → **v3** (son). Bir resmin bütün
ayarlarını ve prompt bloklarını sabitler; sahne yalnızca **poz** ve **mekân**
ekleyebilir, stil kelimesi ekleyemez.

- Sampler: 25 adım, CFG 6, dpmpp_2m / karras.
- Prompt sırası sabit: `tetik, karakter adı, [tek hayalet], karakter gövdesi, [poz], stil, sahne, kuyruk`.
- Negatif: temel liste + karaktere özgü (blob, slime, legs, feet, sharp teeth…) + sahneye özgü.
- Kabul testi: `python scripts/make_keyframes.py --trailer milo_01 --check-lock`
  onaylı Milo tasarımını **piksel piksel** yeniden üretir (1.048.576 pikselin 0'ı farklı).
- v2/v3 farkı: IP-Adapter referansı. v2 arka planlı referans → sahneyi kopyaladı;
  v3 kesilmiş referans + **karakter bölgesi maskesi** (`character_region`,
  IPAdapter `attn_mask`) → arka plan prompt'a uyar, Milo referanstan gelir.

## 5. Karakter: Milo

1. **Tasarım:** SDXL + 80s LoRA, seed 42, 1024×1024, kullanıcı onayladı →
   `benchmarks/references/milo/milo_final_design.png` (provenance JSON'u ile).
2. **Turkuaz gözler:** `spikes/paint_milo_eyes.py` — model yok; göz şekilleri
   flood-fill ile bulunur, içleri turkuaz iris + siyah göz bebeği + beyaz
   parlama ile boyanır → `milo_design_v2.png`.
3. **Kesme:** `spikes/cutout_milo.py` — açık mavi gövde bölgesi 5 başlangıç
   noktasından büyütülür (kontur çizgileri gövdeyi parçalara ayırdığı için),
   delikler doldurulur, düz gri (128) kareye konur → `milo_cutout_v3.png`.
4. **HD:** Real-ESRGAN ile 2048×2048'e büyütülür; büyütmenin konturun
   dışında bıraktığı açık hale, orijinal maske büyütülüp uygulanarak silinir
   → `outputs/hd_plates/milo_cutout_v3_2048_clean.png`.

## 6. Arka planlar

`trailers/milo_03/keyframes.yaml` (+ milo_02'den iki tane), hepsi karaktersiz,
1280×704, 4 seed üretilip en iyisi seçildi:

| Sahne | Dosya (seed) |
|---|---|
| Ay + orman | milo_02 `shot_01_start` seed 45 |
| Ağaç kabuğu (kovuk kodla çizilir) | milo_03 `plate_hollow` seed 42 |
| Orman açıklığı | milo_03 `plate_clearing` seed 42 |
| Yıldızlı gökyüzü + bulutlar | milo_03 `plate_sky` seed 43 |
| Şato + ay (pencere ≈ (505, 500)) | milo_02 `shot_05_plate` seed 42 |

İşleme:
- `scripts/upscale_images.py` → ComfyUI'de ESRGAN + Lanczos → 1920×1056.
- `scripts/depth_maps.py` (kokoro-venv) → 16-bit derinlik PNG (0 uzak, 1 yakın).
- **Ortak renk paleti (grade):** her arka planın kanal ortalaması/sapması %35
  oranında orman arka planına çekilir, sonra ortak gece eğrisi (gölgeler
  koyu maviye, parlak yerler hafif sıcağa).
- Tehlike sahnesi: açıklık %55 karartılır + mor tül.

## 7. Kukla sistemi — `src/ai_studio/puppet.py`

`Rig(çizim, RigSpec, scale)` tek bir çizimi katmanlara ayırır;
`render(Pose, tuval)` bir pozu RGBA resme çevirir.

**RigSpec (2048'lik çizimde, 1024 değerlerinin 2 katı):** kol çokgenleri,
omuz dönme noktaları, kesim çizgileri, göz ve ağız başlangıç noktaları,
gövde rengi örnek noktası.

**Katmanlar:**
- Gövde (kollar kesilmiş, yüz silinmiş).
- İki kol: kolun kendisi + gövdenin derin içinden alınmış kök kısmı;
  gövdenin **arkasına** çizilir, dönünce kök gövdenin altında kalır.
  Kesim çizgisine yakın kontur pikselleri gövde rengine boyanır (dönünce
  çıkıntı kalmasın). Omuzlarda kalan küçük çentikler, sadece pivot
  çevresinde morfolojik kapama ile doldurulup konturlanır (integral-image
  ile hızlı genişletme/daraltma).

**Yüz (her karede kodla çizilir):** gözler (turkuaz iris, siyah kontur,
bakış yönüne kayan göz bebeği, parlama, yukarıdan inen göz kapağı, kapalıyken
mutlu kavis), tek göz kırpma (`wink`), korkuda büyüme (`eye_scale`); ağız:
`smile` (çizimin kendi ağzı), `o` (bağırma), `closed` (gülümseyen çizgi).

**Pose alanları:** konum, boy, `squash` (yassılma/uzama), `tilt`, kol açıları,
`eye_open`, `wink`, `eye_scale`, `gaze`, `mouth`, `drag` (alt gövde geriden
gelir), `wave` + `wave_phase` (kumaş dalgası), `tip` (kafa ucunun salınımı),
`stretch` + `stretch_angle` (hareket yönünde alanı koruyarak esneme), `glow`,
`opacity`.

**Bükülme:** her satır yana kaydırılır: `H·(drag·v² + wave·sin(faz − 4v)·v^1.5) + tip`
(v: tepeden alta 0→1). Esneme: döndür-ölçekle-döndür afin dönüşümü.

Yardımcılar: `ease`, `ease_in`, `ease_out`, `spring`, `bob`, `lerp`.

## 8. Oyunculuk ve zamanlama — `scripts/make_milo_teaser.py`

Dünya koordinatları 1280×704. Yedi sahne (`SHOTS`):

| Sn | Sahne | Hareket |
|---|---|---|
| 0–3 | Ay ve orman | Kamera aydan aşağı iner; sis, yıldızlar |
| 3–6.5 | Kovuk | Karanlıkta gözler açılır, kırpar, sola-sağa bakar, gülümser |
| 6.5–11.5 | Açıklık | Aşağıdan esneyerek fırlar, yaylanarak durur, etrafına bakar; kamera yüzüne yaklaşır ("Meet Milo!"), el sallar, göz kırpar; çömelip (hazırlık) uçar |
| 11.5–14.5 | Gökyüzü | Dalgalı yol + takla; yöne göre esneme, kuyruk, kıvılcım izi |
| 14.5–19 | Tehlike | Mırıldanır; gölge el (kodla çizilmiş örümcek parmaklı siluet) yaklaşır; Milo bakar, çömelir, korkuyla zıplar (kamera sarsılır), hız çizgileri ve iz bırakarak kaçar; el boşluğu kavrar, toz |
| 19–21.5 | Şato | Uzaktan küçülerek pencereye süzülür, "pıt", kıvılcım, pencere ışıldar |
| 21.5–25 | Başlık | Harfler tek tek yaylanır, alt yazı belirir, Milo konar, göz kırpar, iris kapanır |

Animasyon ilkeleri kodda: hazırlık (anticipation), sıkışma-esneme, gecikmeli
takip (follow-through), yaylı duruş (overshoot), yay yolları, bakış ve göz
kırpma ile canlılık.

Aynı dosyada `CUES`: anlatım cümleleri + başlangıç saniyeleri, efekt
zamanları, müzik bölümleri, ortam sesi aralıkları (ses script'i okur).

## 9. HD render — `scripts/make_milo_teaser_hd.py`

`make_milo_teaser.py`'yi modül olarak içe aktarır (oyunculuk aynı kalır),
çıkışı 1920×1056'ya taşır (K = 1.5).

- **Kamera:** zoom + odak noktası; arka planı derinlikle örnekler:
  yakın pikseller kameranın hareketinden %45 daha fazla kayar ve zoom ile
  daha çok büyür → 2.5D. Derinlik yarım çözünürlükte örneklenir.
- **Ön plan (multiplane):** odak dışı koyu dallar (özyinelemeli çizim) ve
  çimen; kameranın 1.6 katı hareket eder.
- **Cel gölge:** siluetin 1/4 boyutlu ondalıklı bulanığından eğim (normal)
  → ışığın tersi tarafında net kenarlı gölge tonu, ışık tarafında parlama,
  ince kenar ışığı; kontur pikselleri korunur. Sahne başına ışık yönü ve
  ortam rengi (tehlikede mor).
- **Hareket bulanıklığı:** hız eşik üstündeyse ±1/72 sn'lik 3 poz ortalanır.
- **Atmosfer:** kayan sis (düşük frekanslı gürültü), parıldayan yıldızlar,
  ateş böcekleri, ay ışınları, yere vuran hayalet parıltısı, bloom.
- **Film görünümü:** her karede gren, iki karede bir 1 px görüntü kayması,
  kenar kararması; sahne geçişlerinde 4 karelik kararma; sonda iris.
- **Hız:** kare başına ~1.6 sn; 600 kare ~17 dk. ffmpeg'e ham RGB akışı,
  libx264 CRF 15.

## 10. Ses — `scripts/make_milo_audio.py` (+ `scripts/kokoro_tts.py`)

- **Anlatım:** Kokoro (`am_michael`, 0.92 hız). Bir cümle sonrakine
  yetişmiyorsa `atempo` ile perdeyi bozmadan hızlandırılır (en fazla ~1.16×
  oldu). Fragman yankısı (`aecho`) + kompresör.
- **Efektler (numpy sentezi):** baykuş, ışıltı, vınlama, yay sesi, korku
  vurgusu (uyumsuz akor), "uh-oh" düdüğü, kaçış (kayan düdük + gürültü),
  kavrama, pıt, çan, ding, takla düdüğü, iris "whoop".
- **Ortam:** rüzgâr (modülasyonlu filtreli gürültü), cırcır böceği cıvıltıları.
- **Müzik (numpy sentezi), bölümler:** gizem (La minör drone + arpej) →
  büyü (pad, çanlar, Do majör arpej, bas) → tehlike (132 BPM synthwave:
  kick, snare, hi-hat, bas arpej) → kahramanlık (Re majör akor + arpej +
  davul) → kapanış (akor + çan, sönme).
- **Miks:** anlatım sırasında müzik otomatik kısılır (zarf izleyici), limiter,
  `loudnorm` −16 LUFS / −1.5 dBTP, AAC 192k.

## 11. Yeniden üretme

```bash
# (bir kez) arka planlar ve büyütme/derinlik
python scripts/make_keyframes.py --trailer milo_03
python scripts/upscale_images.py --width 1920 --height 1056 --out outputs/hd_plates <5 arka plan>
python scripts/upscale_images.py --width 2048 --height 2048 --out outputs/hd_plates benchmarks/references/milo/milo_cutout_v3.png
~/ai-tools/kokoro-venv/Scripts/python.exe -I scripts/depth_maps.py outputs/hd_plates/depth outputs/hd_plates/*_1920x1056.png
#   (+ milo_cutout_v3_2048_clean.png: büyütülmüş çizime orijinal maske uygulanır, bkz. §5)

# film
python scripts/make_milo_teaser_hd.py            # --preview: hızlı kontak sayfası; --only 6.5 11.5: tek sahne
python scripts/make_milo_audio.py --video outputs/trailers/milo_03_hd/<zaman>/teaser.mp4 --tag final
```

Çıktılar asla üzerine yazılmaz; her çalıştırma yeni zaman damgalı klasör açar.

## 12. Kalite kontrol

- `scripts/check_clip.py`: renk dalgalanması (arka plan renginin kareden kareye
  pompalanması), ani sıçrama, karakter alanının ani/büyük değişimi. Eşikler
  milo_01'deki gerçek hatalarla ayarlandı; ölçülebilen 5 hatanın 5'ini yakaladı.
  Anlam hatalarını (fazla el, saçma sahne) **yakalayamaz**.
- Her çekimin karelerine görsel bakış (kontak sayfaları) zorunlu adım.
- Birim testleri: `tests/test_keyframes.py` (kilit, plan, prompt sırası,
  workflow, maske) — `pytest` 78/78 geçiyor (keyframe + trailer testleri).

## 13. Bu noktaya nasıl gelindi (dersler)

1. Wan 2.2 Turbo ile video: karakter tutarsız, bacak/el çıktı, kamera kaydı;
   prompt'a "static camera" yazmak etkisiz (CFG 1).
2. IP-Adapter referansı arka planla verilince sahneyi kopyalar → kesilmiş
   referans + bölge maskesi gerekir.
3. "Sis" gibi şekilsiz kelimeler video modelinde nesneye dönüşür (pembe balon).
4. Yapay zekâ hareketi yerine kod ile cel/kukla animasyonu: karakter hiç
   bozulmaz, hareket planlı olur; kalite sıçraması burada geldi.
5. Cel gölgede 8-bit bulanık maske dalgalı desen üretir → ondalıklı hesap.

## 14. Sınırlar

- Milo tek bir **önden** çizimden türetilir: yan profil, 3/4, arkasını dönme,
  konuşma ağız senkronu yok. Daha zengin oyunculuk için farklı açılardan poz
  çizimleri (ve onların tutarlılığı) gerekir.
- Gölge el ve kovuk kodla çizilmiş basit şekiller.
- Müzik synth ile sentezleniyor; orkestral değil (MusicGen düşünülüyor,
  CC-BY-NC lisanslı).
- `outputs/` (üretilen videolar, ara görseller) git dışında.

## 15. Dosya haritası

| Dosya | Görev |
|---|---|
| `configs/style_locks/milo_80s_v{1,2,3}.yaml` | Kilitli stil |
| `trailers/milo_01..03/` | Shot list'ler ve keyframe planları |
| `src/ai_studio/keyframes/` | Kilit, plan, ComfyUI keyframe üretimi |
| `scripts/make_keyframes.py` | Keyframe CLI (`--dry-run`, `--check-lock`) |
| `src/ai_studio/puppet.py` | Kukla sistemi |
| `scripts/make_milo_teaser.py` | Oyunculuk, zamanlama, cue'lar (1280×704 sürüm de üretir) |
| `scripts/make_milo_teaser_hd.py` | HD render |
| `scripts/upscale_images.py`, `scripts/depth_maps.py` | Büyütme, derinlik |
| `scripts/make_milo_audio.py`, `scripts/kokoro_tts.py` | Ses |
| `scripts/make_title_card.py` | Logo/başlık yazısı (parlama) |
| `scripts/check_clip.py` | Otomatik klip kontrolü |
| `spikes/paint_milo_eyes.py`, `spikes/cutout_milo.py` | Göz boyama, kesme |
| `benchmarks/references/milo/` | Onaylı tasarım, turkuaz gözlü sürüm, kesilmiş referans (provenance'lı) |
| `outputs/trailers/milo_03_hd/20261006T224915/audio_final/teaser_with_audio.mp4` | Son film (= Masaüstü `milo_teaser_hd.mp4`) |
