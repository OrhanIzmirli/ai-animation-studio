# Image backends for character sheet + per-shot keyframes

Date read: 2026-10-05

Prices, limits, model IDs and licenses change often. Every figure below was read on the date above from the linked page; re-check before spending money or shipping.

Conventions:

- A claim is followed by an inline link to the page it was read from.
- "UNVERIFIED:" marks anything that could not be confirmed from a page that was actually opened during this research. Nothing marked UNVERIFIED should be relied on.
- "Secondary" marks blogs / forum posts (used only for VRAM and speed reports).
- "Derived" marks simple arithmetic done on cited numbers (not a quote from a source).
- No model was run, no API was called, nothing was downloaded. All fit/speed statements are from documents, not from measurement on the project's hardware.

Use case recap (from the task): one original character (young man, dark hooded streetwear, graphic-novel style), one canonical reference image of 1672x941 px (Derived: ratio 1.777, i.e. 16:9), needed outputs are a 5-view character sheet and per-shot keyframes that later condition an image-to-video model; outputs are used commercially.

---

## 1. Google Gemini image models ("Nano Banana" family) - Gemini API / Vertex AI

Current line-up per the official guide: Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`), Nano Banana 2 (`gemini-3.1-flash-image`), Nano Banana Pro (`gemini-3-pro-image`), and the legacy Nano Banana (`gemini-2.5-flash-image`) ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).

IMPORTANT: `gemini-2.5-flash-image` is listed with shutdown date October 2, 2026, i.e. three days before the date read ([deprecations](https://ai.google.dev/gemini-api/docs/deprecations)); the pricing page says it "will be shut down on October 2, 2026; migrate to Gemini 3.1 Flash Image or Gemini 3.1 Flash Lite Image" ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). UNVERIFIED: whether the endpoint actually stopped answering on that date (no API call was made).

### Price per image and free tier

All figures from the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing) (paid tier, USD):

| Model | Input (per 1M tokens) | Image output | Batch image output |
|---|---|---|---|
| `gemini-3.1-flash-image` | $0.50 (text/image) | $60 per 1M tokens = $0.045 per 0.5K image, $0.067 per 1K, $0.101 per 2K, $0.151 per 4K | $0.022 / $0.034 / $0.050 / $0.076 |
| `gemini-3.1-flash-lite-image` | $0.25 (text/image/video) | $30 per 1M tokens = $0.0336 per 1K image | $0.0168 per 1K image |
| `gemini-3-pro-image` | $2.00 (text/image), "equivalent to $0.0011 per image" | $120 per 1M tokens = $0.134 per 1K/2K image, $0.24 per 4K image | $0.067 per 1K/2K, $0.12 per 4K |
| `gemini-2.5-flash-image` (deprecated) | $0.30 (text/image) | $0.039 per image | $0.0195 per image |

- Free tier (API): the "Free Tier" column reads "Not available" for input and output of all four image models ([pricing](https://ai.google.dev/gemini-api/docs/pricing)).
- Text and thinking output is billed separately ($3 per 1M for 3.1 Flash Image, $12 per 1M for 3 Pro Image) ([pricing](https://ai.google.dev/gemini-api/docs/pricing)); the guide states "thinking tokens are billed by default for thinking models" and that Gemini 3 image models' thinking "cannot be disabled in the API" ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- Vertex AI: the Vertex pricing page lists Gemini 3.1 Flash Image at $0.50 input and $60.00 image output per 1M tokens (Global) and $0.55 / $66 (non-global), and Gemini 3 Pro Image at $2.00 input and $120.00 image output (Global) ([Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)).

### Commercial-use terms of outputs; watermarking / provenance

- Ownership: "Some of our Services allow you to generate original content. Google won't claim ownership over that content. You acknowledge that Google may generate the same or similar content for others and that we reserve all rights to do so." ([Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms), effective March 23, 2026).
- Intended audience: the terms state that use of Google AI Studio and the Gemini API "is for developers building with Google AI models for professional or business purposes, not for consumer use", and users must be 18 or older ([terms](https://ai.google.dev/gemini-api/terms)).
- The terms contain a disclaimer to "use discretion before relying on, publishing, or otherwise using content" provided by the services ([terms](https://ai.google.dev/gemini-api/terms)).
- Watermark: "All generated images include a SynthID watermark" ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)). There is no documented switch to disable it on that page.
- UNVERIFIED: whether Gemini API image outputs also carry C2PA metadata (the image generation guide does not mention C2PA; no other Google page on this was opened).
- UNVERIFIED: Vertex AI output-ownership / indemnity terms (Google Cloud service-specific terms were not opened).

### Identity-consistency features

- Reference image counts per model ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)):
  - `gemini-3.1-flash-image`: up to 10 object images "with high-fidelity" plus "up to 4 images of characters to maintain character consistency"; total up to 14 reference images.
  - `gemini-3-pro-image`: up to 6 object images, "up to 5 images of characters to maintain character consistency", "up to 3 images to be used as style references"; up to 14 images in total.
  - `gemini-3.1-flash-lite-image`: up to 14 object images; character and style rows are "N/A"; described as "Not optimized for multiple reference inputs or multi-turn sequential editing".
  - `gemini-2.5-flash-image`: "works best with up to 3 images as input".
- Documented character-sheet behaviour: the guide has a section "Character consistency: 360 view": "You can generate 360-degree views of a character by iteratively prompting for different angles. For best results, include previously generated images in subsequent prompts to maintain consistency. For complex poses, include a reference image of the selected pose." ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- Edit in place: text-and-image-to-image editing and multi-turn editing are documented; "Inpainting (semantic masking)" is described as "Conversationally define a 'mask' to edit a specific part of an image while leaving the rest untouched" ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)). No pixel-mask input parameter is documented on that page.
- Detail preservation is prompt-driven: "To ensure critical details (like a face or logo) are preserved during an edit, describe them in great detail along with your edit request." ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- A sequential-art example ("Make a 3 panel comic in a gritty, noir art style...") built on an input character image is included in the guide ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).

### API shape

- The guide's code samples use the Interactions API: `client.interactions.create(model="gemini-3.1-flash-image", input=...)`, with input images passed inline as base64 parts (`{"type": "image", "mime_type": "image/png", "data": ...}`) and multi-turn editing via a previous interaction id ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- Output control: `response_format` with `type: "image"`, `aspect_ratio`, `image_size`, `mime_type` (examples use `image/jpeg` and `image/png`); "By default, the model matches the output image size to that of your input image, or otherwise generates 1:1 squares." ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- Sizes for 16:9 ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)):
  - 3.1 Flash Image: 688x384 (0.5K), 1376x768 (1K), 2752x1536 (2K), 5504x3072 (4K). Aspect ratios: 1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9.
  - 3 Pro Image (the table is headed "3.1 Pro Image" in the guide): 1376x768 (1K), 2752x1536 (2K), 5504x3072 (4K). Aspect ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9.
  - 3.1 Flash Lite Image: 1K only.
  - 2.5 Flash Image: 1344x768 for 16:9, single ~1K size.
- Seed: the image generation guide does not mention a seed. The Interactions API reference lists `generation_config.seed`: "Seed used in decoding for reproducibility" ([Interactions API reference](https://ai.google.dev/api/interactions-api)). UNVERIFIED: that this seed makes image output reproducible / deterministic for the image models (no statement found; not tested).
- Number of outputs: "The model won't always follow the exact number of image outputs that the user explicitly asks for." ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- Batch: image generation "can also be run as batch jobs using the Batch API ... You get higher rate limits in exchange for a turnaround of up to 24 hours" ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)); Batch is priced at 50% of standard cost and the Batch page notes "This feature is currently only available with the generateContent API" ([Batch API](https://ai.google.dev/gemini-api/docs/batch-api)). Note the tension: the image guide's samples use the Interactions API while Batch is documented for generateContent only.
- Model limits for `gemini-3.1-flash-image`: input token limit 131,072, output token limit 32,768, inputs text/image/video/PDF, Batch API supported ([model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image)).

### Rate limits

- Numeric RPM/TPM/RPD/IPM values are not printed in the docs; the page says limits "can be viewed in Google AI Studio" and "Specified rate limits are not guaranteed and actual capacity may vary" ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)). UNVERIFIED: the actual per-model numbers (the AI Studio rate-limit page needs a signed-in account; not opened).
- Images per minute (IPM) is a dimension applied only to image-generating models ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).
- Tiers: Free (active project); Tier 1 (billing account linked, $250 billing tier cap); Tier 2 (paid $100 + 3 days, $2,000 cap); Tier 3 (paid $1,000 + 30 days, $20,000 - $100,000+ cap) ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).
- Spend-rate limit per rolling 10 minutes: Tier 1 $10, Tier 2 $50, Tier 3 $200 ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).
- Batch enqueued-token limits, Tier 1: Gemini 3.1 Flash Image Preview 1,000,000; Gemini 3.1 Flash Lite Image 2,000,000; Gemini 3 Pro Image Preview 2,000,000 (the table still uses the "Preview" names) ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).
- Since the image models have no free tier (see above), free-tier rate limits do not apply to them ([pricing](https://ai.google.dev/gemini-api/docs/pricing)).

### Risks

- Deprecation cadence: `gemini-2.5-flash-image` released October 2, 2025, shutdown October 2, 2026; `gemini-3-pro-image-preview` (Nov 20, 2025) and `gemini-3.1-flash-image-preview` (Feb 26, 2026) both shut down June 25, 2026; `gemini-3.1-flash-image` and `gemini-3-pro-image` (both released May 28, 2026) have "No shutdown date announced"; Imagen 4 models shut down August 17, 2026 ([deprecations](https://ai.google.dev/gemini-api/docs/deprecations)). A pipeline pinned to one model ID can therefore lose its backend within about a year.
- Source conflict: the deprecations table names `gemini-3.1-flash-image-preview` as replacement for `gemini-2.5-flash-image`, although that preview ID is itself listed as shut down on June 25, 2026 ([deprecations](https://ai.google.dev/gemini-api/docs/deprecations)); the pricing page instead says to migrate to Gemini 3.1 Flash Image or Flash Lite Image ([pricing](https://ai.google.dev/gemini-api/docs/pricing)).
- Data use: paid tier "Used to improve our products: No"; free tier "Yes" ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). For unpaid services "human reviewers may read, annotate, and process your API input and output"; for paid services "Google doesn't use your prompts ... or responses to improve our products" and "logs prompts and responses for a limited period of time" for abuse detection ([terms](https://ai.google.dev/gemini-api/terms)).
- Retention: "Google retains the following data for fifty-five (55) days for the purposes of detecting and preventing violations" ([usage policies](https://ai.google.dev/gemini-api/docs/usage-policies)).
- Region: in the EEA, Switzerland or UK only Paid Services may be used when making API clients available to users ([terms](https://ai.google.dev/gemini-api/terms)).
- Policy refusals: the guide reminds users to hold rights to uploaded images and points to the Prohibited Use Policy ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)). UNVERIFIED: concrete refusal behaviour for stylised drawings of a fictional adult male (the Prohibited Use Policy page was not opened; nothing tested).
- Invisible watermark on every output (SynthID), no documented opt-out ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- Determinism: no documented reproducibility guarantee for images (see Seed above).
- Cost creep: thinking is always on and billed for Gemini 3 image models ([image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)).
- UNVERIFIED: quality for graphic-novel line art specifically (the guide shows a noir comic example, but no measurement or evaluation was found or performed).

---

## 2. OpenAI GPT Image - API

Current line-up per the official guide: `gpt-image-2.5-sunburst` ("for workflows where editing precision matters most") and `gpt-image-2.5-flare` ("fast, high-quality everyday image generation"); earlier models `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini` are documented under "Earlier GPT Image models" ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)). Default snapshots: `gpt-image-2.5-sunburst-2026-09-08` ([model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst)), `gpt-image-2.5-flare-2026-09-08` ([model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare)), `gpt-image-2-2026-04-21` ([model page](https://developers.openai.com/api/docs/models/gpt-image-2)).

### Price per image and free tier

Token prices per 1M tokens ([pricing](https://developers.openai.com/api/docs/pricing)):

| Model | Text input | Image input | Cached image input | Image output | Batch image output |
|---|---|---|---|---|---|
| `gpt-image-2.5-sunburst` / `gpt-image-2.5-flare` | $5.00 | $8.00 | $2.00 | $30.00 | $15.00 |
| `gpt-image-2` | $5.00 | $8.00 | $2.00 | $30.00 | $15.00 |
| `gpt-image-1.5` | $5.00 | $8.00 | $2.00 | $32.00 | $16.00 |
| `gpt-image-1` | $5.00 | $10.00 | $2.50 | $40.00 | $20.00 |
| `gpt-image-1-mini` | $2.00 | $2.50 | $0.25 | $8.00 | $4.00 |

Per-image output prices published in the guide ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)):

| Model | Quality | 1024x1024 | 1024x1536 | 1536x1024 |
|---|---|---|---|---|
| GPT Image 2 | Low / Medium / High | $0.006 / $0.053 / $0.211 | $0.005 / $0.041 / $0.165 | $0.005 / $0.041 / $0.165 |
| GPT Image 1.5 | Low / Medium / High | $0.009 / $0.034 / $0.133 | $0.013 / $0.05 / $0.2 | $0.013 / $0.05 / $0.2 |
| GPT Image 1 | Low / Medium / High | $0.011 / $0.042 / $0.167 | $0.016 / $0.063 / $0.25 | $0.016 / $0.063 / $0.25 |
| GPT Image 1 Mini | Low / Medium / High | $0.005 / $0.011 / $0.036 | $0.006 / $0.015 / $0.052 | $0.006 / $0.015 / $0.052 |

- UNVERIFIED: per-image dollar prices for `gpt-image-2.5-sunburst` / `-flare`. The guide only offers an interactive calculator for these and states "Equal token rates don't mean equal cost per image: token consumption can differ by model and quality setting" ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)); the calculator values could not be read from the page source.
- Input images are billed as image input tokens on top of output; for `gpt-image-2` "image input tokens can be higher for edit requests that include reference images" because inputs are always processed at high fidelity ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- Free tier: the rate-limit tables on the model pages start at Tier 1 and have no Free row ([gpt-image-2.5-sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst), [gpt-image-2](https://developers.openai.com/api/docs/models/gpt-image-2)). UNVERIFIED: an explicit statement that GPT Image models are unavailable on the free API tier (inferred from the missing row only).

### Commercial-use terms of outputs; watermarking / provenance

- Ownership: "As between Customer and OpenAI, to the extent permitted by applicable law, Customer: (a) retains all ownership rights in Input; and (b) owns all Output. OpenAI hereby assigns to Customer all OpenAI's right, title, and interest, if any, in and to Output." (section 4.1, [OpenAI Services Agreement](https://openai.com/en-GB/policies/services-agreement/), updated 1 December 2025, effective January 1, 2026).
- Similarity: "Output may not be unique, and other users may receive similar content from OpenAI's services." (section 4.4, [Services Agreement](https://openai.com/en-GB/policies/services-agreement/)).
- Indemnity: OpenAI agrees to indemnify against third-party claims "alleging that the Services infringe any third-party IP Right", with exclusions including Customer Content and combination with non-OpenAI products (section 13.1, [Services Agreement](https://openai.com/en-GB/policies/services-agreement/)). UNVERIFIED: the full scope of that clause (only the opening of section 13.1 was read).
- Provenance: "Supported images generated with ChatGPT, Codex, and the OpenAI API include both signals" (C2PA metadata and SynthID watermarks); C2PA Content Credentials "can sometimes be removed by platforms, editing tools, or file conversions" ([OpenAI Help Center: provenance signals](https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images)).

### Identity-consistency features

- Multi-image reference: the edits endpoint accepts multiple input images; "For GPT image models, you can provide up to 16 images." ([Images API reference](https://developers.openai.com/api/reference/resources/images)). The guide's example uses 4 reference images ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- Mask editing: supported; image and mask "must be of the same format and size (less than 50MB in size)", the mask needs an alpha channel, and "If you provide multiple input images, the mask will be applied to the first image." "Masking with GPT Image is entirely prompt-based. The model uses the mask as guidance, but may not follow its exact shape with complete precision." ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- `input_fidelity` ("Control how much effort the model will exert to match the style and features, especially facial features, of input images"): supports `high` and `low` on `gpt-image-1` and `gpt-image-1.5`, `low` only on `gpt-image-1-mini`, and must be omitted for `gpt-image-2`, which "processes every image input at high fidelity automatically" ([Images API reference](https://developers.openai.com/api/reference/resources/images), [image generation guide](https://developers.openai.com/api/docs/guides/image-generation)). UNVERIFIED: how `input_fidelity` behaves on the two `gpt-image-2.5` models (not stated in the pages read).
- Multi-turn editing via the Responses API image generation tool ("Iteratively make high fidelity edits to images with prompting") ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- Documented limitation: "Consistency: While capable of producing consistent imagery, the model may occasionally struggle to maintain visual consistency for recurring characters or brand elements across multiple generations." ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).

### API shape

- Endpoints: `v1/images/generations`, `v1/images/edits`, Batch `v1/batch` (all "Supported" for the 2.5 models); the model is not callable directly on `v1/responses` but is used there as the `image_generation` tool with a mainline model ([model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst), [image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- SDK call (Python, from the guide): `client.images.generate(model="gpt-image-2.5-sunburst", prompt=prompt)`; edits use multipart `image[]=@file.png` fields; the Responses API accepts input images as URL, base64 data URL, or Files API file ID ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation), [Images API reference](https://developers.openai.com/api/reference/resources/images)).
- Sizes: recommended `1024x1024`, `1536x1024`, `1024x1536`; custom `WIDTHxHEIGHT` such as `1536x864` allowed if both edges are multiples of 16, aspect ratio between 1:3 and 3:1, no edge above 3840 px, total pixels between 655,360 and 8,294,400; "Resolutions above 2560x1440 are experimental." ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- Quality: `low`, `medium`, `high`, `xhigh`, `max`, `auto` (2.5 models; earlier models up to `high`) ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- Output format: `png` (default), `jpeg`, `webp`; transparent background with png/webp ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- Multiple images: "You can set the `n` parameter to generate multiple images at once in a single request" ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).
- Seed: no `seed` parameter appears in the Images API reference for create / edit ([Images API reference](https://developers.openai.com/api/reference/resources/images)) or in the image generation guide ([guide](https://developers.openai.com/api/docs/guides/image-generation)); no determinism guarantee is documented. UNVERIFIED: whether any undocumented seed exists.
- Batch: Batch API supported, at half the image-output token price ($15 vs $30 per 1M) ([model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst), [pricing](https://developers.openai.com/api/docs/pricing)).
- Latency: "Complex prompts may take up to 2 minutes to process." ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)).

### Rate limits

`gpt-image-2.5-sunburst` and `gpt-image-2` publish the same table ([gpt-image-2.5-sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst), [gpt-image-2](https://developers.openai.com/api/docs/models/gpt-image-2)):

| Tier | TPM | IPM (images per minute) |
|---|---|---|
| Tier 1 | 100,000 | 5 |
| Tier 2 | 250,000 | 20 |
| Tier 3 | 800,000 | 50 |
| Tier 4 | 3,000,000 | 150 |
| Tier 5 | 8,000,000 | 250 |

### Risks

- Deprecations ([deprecations](https://developers.openai.com/api/docs/deprecations)): `gpt-image-1` shuts down October 23, 2026; `gpt-image-1-mini`, `gpt-image-1.5` and `chatgpt-image-latest` shut down Dec 1, 2026; recommended replacement in each case is `gpt-image-2.5-sunburst` or `gpt-image-2.5-flare`. `dall-e-2` and `dall-e-3` were removed 2026-05-12. UNVERIFIED: a shutdown date for `gpt-image-2` or the 2.5 models (none found on the deprecations page).
- Access gate: "you may need to complete the API Organization Verification ... before using GPT Image models" ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)); verification requires an original, physical government-issued ID and, if requested, a selfie ([OpenAI Help Center: verification](https://help.openai.com/en/articles/10910291-api-organization-verification)).
- Moderation: "All prompts and generated images are filtered in accordance with our content policy"; the `moderation` parameter accepts `auto` (default) or `low` ("Less restrictive filtering"); blocked requests return `error.code = "moderation_blocked"` ([image generation guide](https://developers.openai.com/api/docs/guides/image-generation)). UNVERIFIED: concrete refusal rates for a hooded fictional character (not tested). The OpenAI usage policies page returned HTTP 403 and could not be opened.
- Data: "As of March 1, 2023, data sent to the OpenAI API is not used to train or improve OpenAI models (unless you explicitly opt in to share data with us)"; abuse monitoring logs are "retained for up to 30 days" by default; `/v1/images/generations` and `/v1/images/edits` are listed with data used for training "No", abuse monitoring retention "30 days", application state retention "None" ([Data controls](https://developers.openai.com/api/docs/guides/your-data)). Contractually: "OpenAI will not use Customer Content to develop or improve the Services, unless Customer explicitly agrees to such use." (section 4.2, [Services Agreement](https://openai.com/en-GB/policies/services-agreement/)).
- Provenance marks (C2PA + SynthID) are applied to API images ([Help Center](https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images)).
- No seed parameter documented (see API shape), so exact re-generation of a keyframe is not something the documentation promises.
- Recurring-character consistency is a vendor-documented limitation (quoted above).
- UNVERIFIED: quality for graphic-novel line art specifically (no evaluation found or performed).

---

## 3. Qwen-Image-Edit (open weights) on a cloud NVIDIA L4 (24 GB VRAM) + 32 GB system RAM

### Which variants exist and how they are licensed

| Repo | Created | License | Notes |
|---|---|---|---|
| [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | 2025-08-17 | apache-2.0 | original |
| [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | 2025-09-22 | apache-2.0 | multi-image input, native ControlNet conditions |
| [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | 2025-12-17 | apache-2.0 | "mitigate image drift, improved character consistency" |
| [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 2026-09-14 | "qwen-research" (NOT Apache) | unified T2I + editing, 7B, up to 10 reference images |

(Creation dates and license tags from the Hugging Face model API / model cards linked in the table.)

- No `Qwen/Qwen-Image-Edit-2512`, `-2601`, `-2603` or `Qwen-Image-Edit-2.0` repo exists under the Qwen organisation on Hugging Face (API lookups returned not found; org listing checked at [huggingface.co/Qwen](https://huggingface.co/Qwen)). The newest "Edit"-named release is 2511; the newest editing-capable model is Qwen-Image-2.1.
- LICENSE TRAP: Qwen-Image-2.1 is under the "Qwen RESEARCH LICENSE AGREEMENT" (release date September 20, 2026): the grant is "FOR NON-COMMERCIAL PURPOSES ONLY", "Non-Commercial" is defined as "for research or evaluation purposes only", and "You shall not use the Materials for any commercial purpose without obtaining a separate commercial license from us" ([Qwen-Image-2.1 LICENSE](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)). The license restricts use of the "Materials" (weights, code, documentation); it does not contain a clause granting commercial use of outputs. So the Apache-2.0 statement in the task applies to Qwen-Image-Edit / -2509 / -2511 only.

### Price per image and free tier

- No per-image fee: weights are free to download under Apache-2.0 for the three Edit models (license tags on the model cards above). Cost is GPU rental time.
- UNVERIFIED: the hourly price of the project's L4 instance and therefore the cost per image (no L4 speed figure was found, see below; no cloud pricing page was opened).

### Commercial-use terms of outputs; watermarking / provenance

- The model cards state "Qwen-Image is licensed under Apache 2.0." ([Qwen-Image-Edit-2511 card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), [Qwen-Image-Edit-2509 card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)). Apache-2.0 places no restriction on commercial use of the software; the cards contain no separate output-ownership clause.
- The Lightning LoRAs, GGUF conversions and nunchaku quantizations used below are also tagged apache-2.0 on their repos ([lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning), [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF), [QuantStack/Qwen-Image-Edit-2509-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF), [nunchaku-ai/nunchaku-qwen-image-edit-2509](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509)).
- Watermarking: the model cards read mention no SynthID / C2PA / watermark step. UNVERIFIED: absence of any embedded watermark (not tested).

### Identity-consistency features

- 2509: multi-image editing via image concatenation, "person + person", "person + product", "person + scene"; "Optimal performance is currently achieved with 1 to 3 input images"; "Better preservation of facial identity, supporting various portrait styles and pose transformations"; "Native Support for ControlNet: Including depth maps, edge maps, keypoint maps"; a keypoint map can be passed as a second image to change pose; cartoon characters are shown as a supported case ([Qwen-Image-Edit-2509 card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)).
- 2511: "character consistency has been significantly improved. The model can perform imaginative edits based on an input portrait while preserving the identity and visual characteristics of the subject"; improved multi-person consistency; "integrates selected popular LoRAs directly into the base model" ([Qwen-Image-Edit-2511 card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)).
- Mask editing: diffusers documents a Qwen-Image edit inpaint pipeline with a `mask_image` argument ("White pixels in the mask are repainted while black pixels are preserved") ([diffusers QwenImage docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)). UNVERIFIED: that the inpaint pipeline works with the 2509/2511 "Plus" checkpoints specifically.

### API shape (self-hosted)

- diffusers: `QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)`; card example uses `true_cfg_scale=4.0`, `num_inference_steps=40`, `generator=torch.manual_seed(0)`; `image` accepts a single image or a list ([Qwen-Image-Edit-2511 card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), [diffusers QwenImage docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)).
- Seed: `generator` is documented as "One or a list of torch generator(s) to make generation deterministic" ([diffusers QwenImage docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)). UNVERIFIED: bit-exact reproducibility across different GPUs / quantizations (not stated, not tested).
- ComfyUI: native workflow documented for 2511 with `qwen_image_edit_2511_bf16.safetensors`, `qwen_2.5_vl_7b_fp8_scaled.safetensors`, `qwen_image_vae.safetensors` and the optional 4-step Lightning LoRA ([ComfyUI docs: Qwen-Image-Edit-2511](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)).
- GGUF: unsloth's 2511 GGUF repo states it "Uses tooling from ComfyUI-GGUF by city96" ([unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)).
- Few-step: `lightx2v/Qwen-Image-Edit-2511-Lightning` ships 4-step and 8-step LoRAs (bf16 0.85 GB each) and fp8 checkpoints with the LoRA fused; the card claims 4 steps is "~10x faster than standard 40-step inference" ([lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)). 2509 Lightning LoRAs (4/8 steps) are in [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning). diffusers documents loading these LoRAs ([diffusers QwenImage docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)).
- Batch: `num_images_per_prompt` argument exists in the diffusers pipeline ([diffusers QwenImage docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)).

### Does it fit 24 GB VRAM / 32 GB RAM? (file sizes are facts; the fit conclusion is Derived)

File sizes read from the Hugging Face file listings:

| Component | File | Size |
|---|---|---|
| Transformer, bf16 | `qwen_image_edit_2511_bf16.safetensors` ([Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI)) | 40.86 GB |
| Transformer, fp8 | `qwen_image_edit_2511_fp8mixed.safetensors` (same repo) | 20.53 GB |
| Transformer, int8 | `qwen_image_edit_2511_int8_convrot.safetensors` (same repo) | 20.50 GB |
| Transformer, fp8 + 4-step Lightning fused | `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors` ([lightx2v](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)) | 20.58 GB |
| Transformer, GGUF ([unsloth](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)) | Q8_0 / Q6_K / Q5_K_M / Q4_K_M / Q3_K_M / Q2_K | 21.76 / 16.85 / 15.03 / 13.24 / 9.92 / 7.47 GB |
| Transformer, nunchaku 4-bit (2509) ([nunchaku-ai](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509)) | `svdq-int4_r32` / `svdq-int4_r128` | 11.52 / 12.65 GB |
| Text encoder (Qwen2.5-VL 7B) ([Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)) | full / fp8_scaled / nvfp4 | 16.58 / 9.38 / 6.11 GB |
| VAE (same repo) | `qwen_image_vae.safetensors` | 0.25 GB |

- Derived: bf16 transformer (40.86 GB) does not fit 24 GB VRAM and exceeds 32 GB RAM by itself.
- Derived: fp8 transformer (20.5 GB) fits 24 GB VRAM alone, but transformer + fp8 text encoder (20.5 + 9.4 = 29.9 GB) does not fit in VRAM together, and 29.9 GB is also close to the 32 GB system RAM. The text encoder therefore has to be unloaded / offloaded between encode and sampling, and RAM headroom is thin.
- Derived: GGUF Q5_K_M (15.0 GB) or Q4_K_M (13.2 GB) plus the fp8 text encoder (9.4 GB) sums to 24.4 / 22.6 GB, i.e. at or below the 24 GB limit before activations; these and the nunchaku int4 files (11.5-12.7 GB) are the variants with real headroom on a 24 GB card with 32 GB RAM.
- nunchaku: `INT4` is for "non-Blackwell GPUs (pre-50-series)", `NVFP4` for 50-series; r32 "for faster inference", r128 "for better quality but slower inference"; 4-step and 8-step Lightning-fused variants exist; official nunchaku repos exist for Qwen-Image-Edit and -2509 ([nunchaku-ai/nunchaku-qwen-image-edit-2509](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509), [nunchaku README](https://github.com/nunchaku-ai/nunchaku)). For 2511 only third-party nunchaku uploads were found (for example [QuantFunc/Nunchaku-Qwen-Image-EDIT-2511](https://huggingface.co/QuantFunc/Nunchaku-Qwen-Image-EDIT-2511)); UNVERIFIED: their quality and provenance.
- Secondary (blog, not a measurement table): fp8 is described as the 24 GB option and Q4_K_M GGUF as a ~14 GB option, with BF16 needing 48 GB+ VRAM and 64 GB RAM ([Thunder Compute blog, June 9, 2026](https://www.thundercompute.com/blog/qwen-image-edit-comfyui)).
- Secondary, conflicting: another blog lists "FP8 ~15GB" and "BF16 ~30GB", which contradicts the Hugging Face file sizes above (20.5 GB and 40.9 GB); its author states the figures are research-based, not measured ([lilting.ch, January 14, 2026](https://lilting.ch/en/articles/qwen-image-edit-2511-local-specs)). The Hugging Face file sizes are the primary source.

### Expected speed

- UNVERIFIED: speed of any Qwen-Image-Edit variant on an NVIDIA L4. No source reporting L4 timings was found.
- Primary, other hardware: diffusers docs report `torch.compile` speedup "A100 80GB: 4.70s -> 1.93s" for a Qwen-Image pipeline step configuration ([diffusers QwenImage docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)).
- Primary claim: 4-step Lightning is "~10x faster than standard 40-step inference" ([lightx2v card](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)).
- Secondary, different model and hardware (for scale only): Qwen-Image-2.1 in bf16 on a DGX Spark: 54.1 s per 1024x1024 image at 40 steps with 31.6 GB peak memory; with reference images at 1024x1024 / 40 steps: 62.5 s (1 reference), 184 s (3 references), 598 s (10 references) ([NVIDIA developer forum post, September 30, 2026](https://forums.developer.nvidia.com/t/qwen-image-2-1-on-dgx-spark-54-s-per-1024-image-40-steps-31-6-gb-and-a-fix-for-the-pink-vertical-line/384774)).

### Rate limits

- None (self-hosted); throughput is bounded by the GPU. Queue / instance limits of the cloud provider: UNVERIFIED (not researched).

### Risks

- OOM / RAM pressure: see the Derived fit notes; 32 GB RAM is close to the combined size of fp8 transformer + fp8 text encoder.
- Lightning LoRAs run at CFG 1.0 per the secondary guide ([Thunder Compute blog](https://www.thundercompute.com/blog/qwen-image-edit-comfyui)), which removes negative-prompt control. UNVERIFIED: effect on comic line quality.
- "Optimal performance is currently achieved with 1 to 3 input images" for 2509 ([card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)); UNVERIFIED: the corresponding limit for 2511 (the 2511 card read does not restate a number).
- Upgrade trap: the successor (Qwen-Image-2.1) is research / non-commercial licensed ([LICENSE](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)), so "latest Qwen" is not automatically usable for the trailer.
- No vendor deprecation (weights are local) and no data leaves the instance beyond what the cloud host sees.
- Low-bit quantization quality loss for clean line art: UNVERIFIED (no evaluation found).

---

## 4. What realistically fits a laptop RTX 4060 (8 GB VRAM) + 16 GB system RAM

All "fit" statements here are Derived from published file sizes and vendor VRAM claims unless marked Secondary. Nothing was run. A general constraint (Derived): whatever does not fit in 8 GB VRAM must sit in the 16 GB system RAM together with the OS and ComfyUI, so transformer + text encoder files summing to much more than about 16 GB imply swapping to disk.

### 4a. FLUX.2 [klein] 4B (Black Forest Labs) - Apache-2.0

- License: apache-2.0; "Open weights available for commercial use under the Apache 2.0 license" ([FLUX.2-klein-4B card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)). The 9B klein models are NOT Apache: "4B models are Apache 2.0. 9B models use the FLUX.2-dev Non-Commercial License" ([flux2 GitHub README](https://github.com/black-forest-labs/flux2)).
- Price / free tier: no per-image fee for local use (Apache-2.0 weights).
- Identity features: "Text-to-image and image-to-image multi-reference editing in a single unified model"; step-distilled (4 steps, `guidance_scale=1.0` in the card example) ([card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)); multi-reference for [klein] is "Up to 4" images ([BFL docs: FLUX.2 overview](https://docs.bfl.ai/flux_2/flux2_overview)). UNVERIFIED: mask-based editing support for klein (not found in the pages read).
- API shape: diffusers `Flux2KleinPipeline`, seed via `generator=torch.Generator(device=device).manual_seed(0)` ([card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)); ComfyUI native workflow with `flux-2-klein-4b-fp8.safetensors`, `qwen_3_4b.safetensors`, `flux2-vae.safetensors` ([ComfyUI docs: FLUX.2 klein](https://docs.comfy.org/tutorials/flux/flux-2-klein)).
- File sizes: transformer bf16 7.75 GB ([card repo](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)), fp8 4.07 GB ([FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8)), GGUF Q8_0 4.30 GB / Q4_K_M 2.60 GB ([unsloth/FLUX.2-klein-4B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF)); text encoder in the diffusers repo is 4.97 + 3.08 GB ([card repo](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)).
- VRAM claims CONFLICT between official sources: "fits in ~13GB VRAM and is accessible on NVIDIA RTX 3090/4070 and above" ([HF card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), also [BFL docs](https://docs.bfl.ai/flux_2/flux2_overview)) versus "Klein 4B fits in ~8GB VRAM (RTX 3090/4070 and up)" ([flux2 GitHub README](https://github.com/black-forest-labs/flux2)) versus "distilled ~1.2s (5090) - 8.4GB VRAM" ([ComfyUI docs](https://docs.comfy.org/tutorials/flux/flux-2-klein)). BFL also states a target of "sub-5-second image generation on an 8GB VRAM laptop" for an ASUS-optimised build ([BFL blog, June 4, 2026](https://bfl.ai/blog/flux2-klein-on-device)).
- Derived fit: fp8 transformer (4.07 GB) or Q4-Q8 GGUF (2.6-4.3 GB) fits 8 GB VRAM for the denoising pass; the ~8 GB text encoder cannot be resident in VRAM at the same time and must be offloaded to RAM (8 GB of the 16 GB) or replaced by a quantized encoder.
- Secondary speed: "FLUX.2 Klein 4B (GGUF ~2.6GB, 4 steps, ~15-30s/image)" on 8 GB cards (RTX 4060 / 3060 Ti reference); the page does not clearly state the numbers were measured by its author ([localaimaster.com, updated August 2026](https://localaimaster.com/vram/best-flux-model-8gb-vram)).
- Risks: only 4 reference images; safety filters are "encouraged" for 4B (required for 9B) ([card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)); the reference inference code "implements an example of pixel-layer watermarking" ([card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)), UNVERIFIED whether diffusers / ComfyUI apply it; identity fidelity of a 4B distilled model for a 5-view sheet: UNVERIFIED (no evaluation found).

### 4b. FLUX.1 Kontext [dev] (quantized) - NON-commercial license

- License: FLUX.1 [dev] Non-Commercial License v1.1.1, which explicitly covers "FLUX.1 Kontext [dev]" ([license text](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)); the Hugging Face repo and its GGUF / nunchaku derivatives carry `flux-1-dev-non-commercial-license` ([FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), [QuantStack GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF), [nunchaku](https://huggingface.co/nunchaku-ai/nunchaku-flux.1-kontext-dev)).
- Exact terms (two clauses that pull in different directions, both from the [license text](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)):
  - Outputs: "We claim no ownership rights in and to the Outputs. ... You may use Output for any purpose (including for commercial purposes), except as expressly prohibited herein. You may not use the Output to train, fine-tune or distill a model that is competitive with the FLUX.1 [dev] Model or the FLUX.1 Kontext [dev] Model."
  - Use of the model: "You may only access, use, Distribute, or create Derivatives of the FLUX.1 [dev] Model or Derivatives for Non-Commercial Purposes." "Non-Commercial Purpose" applies "only so far as you do not receive any direct or indirect payment arising from the use of the FLUX.1 [dev] Model", and "use (a) for revenue-generating activity, (b) in direct interactions with or that has impact on end users, or (c) to train, fine tune or distill other models for commercial use, in each case is not a Non-Commercial Purpose."
  - The license also conditions access / output creation on implementing content filters or manual review ([license text](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)).
  - Net: the output clause permits commercial use of outputs, but running the model as part of a revenue-generating activity is itself outside the non-commercial grant; a commercial license must be requested from BFL ([license text](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)). UNVERIFIED: how BFL applies this to a solo creator producing a commercial trailer, and the price of a self-hosted commercial license (the [BFL licensing page](https://bfl.ai/licensing) was opened but no price for Kontext [dev] could be read from it).
- File sizes: GGUF Q4_K_S 6.80 GB, Q4_K_M 6.93 GB, Q5_K_S 8.28 GB, Q8_0 12.71 GB ([QuantStack/FLUX.1-Kontext-dev-GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF)).
- Derived fit: Q4 (6.8-6.9 GB) fits 8 GB VRAM with little room for activations; the T5 text encoder must live in system RAM. UNVERIFIED: text-encoder file size and total RAM use (not looked up).
- Identity features / speed on 8 GB: UNVERIFIED (the gated model card could not be read without login; no measured 8 GB Kontext timing was opened).

### 4c. FLUX.1 [schnell] and FLUX.1 [dev] GGUF

- FLUX.1 [schnell]: apache-2.0 ([FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf)); GGUF Q4_K_S 6.78 GB, Q5_K_S 8.26 GB ([city96 GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)). UNVERIFIED: any native reference-image / editing capability (none was found in the pages read; it is a text-to-image model per its HF tags).
- FLUX.1 [dev]: same non-commercial license and same two clauses as 4b ([license text](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)).
- Secondary speed on 8 GB cards: FLUX.1 schnell GGUF Q4_K_S ~6.8 GB, 4 steps, 20-30 s per 1024 px image; FLUX.1 dev GGUF Q4_K_S ~6.8 GB "fits with --lowvram at ~90-150s per 20-step 1024px image"; 12B models with `--lowvram` are said to need 16-32 GB system RAM ([localaimaster.com](https://localaimaster.com/vram/best-flux-model-8gb-vram); not clearly author-measured).

### 4d. Z-Image / Z-Image-Turbo (Tongyi-MAI) - Apache-2.0

- License: apache-2.0 for both ([Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)).
- 6B parameters; Turbo uses 8 NFEs and "fits comfortably within 16G VRAM consumer devices" ([Z-Image-Turbo card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)).
- Identity features: Turbo is a text-to-image pipeline (`ZImagePipeline`); the card lists "Z-Image-Edit" and "Z-Image-Omni-Base" as "To be released" ([Z-Image-Turbo card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)); no `Tongyi-MAI/Z-Image-Edit` repo exists on Hugging Face as of the date read (org listing at [huggingface.co/Tongyi-MAI](https://huggingface.co/Tongyi-MAI)). So no official reference-image editing path was found. The base Z-Image card positions it for "LoRA training, structural conditioning (ControlNet)" and notes "higher variability in composition, facial identity" across seeds ([Z-Image card](https://huggingface.co/Tongyi-MAI/Z-Image)).
- File sizes ([Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)): transformer bf16 12.31 GB, int8 6.20 GB, nvfp4 4.51 GB; text encoder `qwen_3_4b` 8.04 GB, fp8_mixed 5.63 GB, fp4_mixed 3.48 GB.
- Derived fit: int8 transformer (6.2 GB) fits 8 GB VRAM for denoising with the text encoder offloaded; bf16 (12.3 GB) does not fit in VRAM.
- Speed on 8 GB: UNVERIFIED (no source opened).
- Seed: card example uses `generator=torch.Generator("cuda").manual_seed(42)` ([Z-Image-Turbo card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)).

### 4e. SDXL + IP-Adapter / ControlNet

- SDXL base 1.0 license: CreativeML Open RAIL++-M ([stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)); "Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output you generate and its subsequent uses." ([LICENSE.md](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)).
- IP-Adapter: apache-2.0; SDXL variants include `ip-adapter_sdxl`, `ip-adapter-plus_sdxl_vit-h` ("closer to the reference image") and `ip-adapter-plus-face_sdxl_vit-h` (cropped face as condition) ([h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)).
- License trap: IP-Adapter-FaceID requires InsightFace to extract face embeddings ([h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)); the FaceID repo carries no license tag on Hugging Face. UNVERIFIED: the license of the InsightFace models (not opened).
- Fit on 8 GB / 16 GB, speed, and identity quality: UNVERIFIED (no primary or secondary source with numbers was opened; diffusers only documents `enable_model_cpu_offload()` "for out-of-memory errors" for SDXL ([diffusers SDXL guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl))).
- Seed: standard diffusers `generator` argument (same mechanism as documented for the pipelines above).

### 4f. Qwen-Image-Edit low-bit on 8 GB / 16 GB

- Smallest transformer files: GGUF Q2_K 7.47 GB, Q3_K_S 9.22 GB, Q3_K_M 9.92 GB for 2511 ([unsloth](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)); the smallest text encoder is 6.11 GB (nvfp4) or 9.38 GB (fp8) ([Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)).
- Derived: Q2_K (7.47 GB) barely fits 8 GB VRAM with no room for activations; Q3 and above exceed 8 GB and need partial offload; transformer + text encoder (7.5-9.9 GB + 6.1-9.4 GB = 13.6-19.3 GB) is at or above the 16 GB system RAM. This is the highest OOM / swap risk of the options listed.
- Secondary: "On 8GB, you are limited to Q2_K or Q3_K_S quantizations with the Lightning LoRA enabled." ([Thunder Compute blog](https://www.thundercompute.com/blog/qwen-image-edit-comfyui)).
- nunchaku: docs show per-layer offloading for Qwen-Image that "only requires 3-4GB of VRAM" ([nunchaku docs: Qwen-Image](https://nunchaku.tech/docs/nunchaku/usage/qwen-image.html)), and the README says asynchronous offloading reduces Qwen-Image VRAM use "to as little as 3 GiB" ([nunchaku README](https://github.com/nunchaku-ai/nunchaku)); the int4 r32 Qwen-Image-Edit-2509 file is 11.52 GB ([nunchaku-ai repo](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509)). Derived: 11.5 GB of weights plus a text encoder still has to fit in 16 GB system RAM, so RAM, not VRAM, is the limit. UNVERIFIED: that this works on 16 GB RAM and at what speed.
- License: apache-2.0 (as in section 3).

### 4g. Not usable commercially without a separate license (listed for completeness)

- FLUX.2 [dev], FLUX.2 [klein] 9B, 9B KV and 9B Base: FLUX Non-Commercial License v2.1 ([flux2 GitHub README](https://github.com/black-forest-labs/flux2), [license text](https://github.com/black-forest-labs/flux2/blob/main/model_licenses/LICENSE-FLUX-NON-COMMERICAL)); it has the same structure as 4b: "You may use Output for any purpose (including for commercial purposes), except as expressly prohibited herein", while model use is limited to Non-Commercial Purposes.
- Qwen-Image-2.1: Qwen Research License, non-commercial ([LICENSE](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)); also a 33 GB checkpoint with 31.6 GB peak memory in bf16 (Secondary: [NVIDIA forum post](https://forums.developer.nvidia.com/t/qwen-image-2-1-on-dgx-spark-54-s-per-1024-image-40-steps-31-6-gb-and-a-fix-for-the-pink-vertical-line/384774)).

---

## Comparison table

| Option | Price per image / free tier | Commercial use of outputs; watermark | Identity features | API shape; seed | Rate limits | Main risks |
|---|---|---|---|---|---|---|
| Gemini 3.1 Flash Image | $0.067 (1K), $0.101 (2K), $0.151 (4K); batch half; no API free tier | Google "won't claim ownership"; SynthID on all images; C2PA UNVERIFIED | 4 character + 10 object refs; documented 360-view workflow; conversational (semantic) masking only | Interactions API, base64 input; 16:9 at 1376x768 / 2752x1536 / 5504x3072; seed field exists, image determinism UNVERIFIED | Numbers only in AI Studio (UNVERIFIED); spend cap $10/10 min at Tier 1 | ~1-year model lifetimes; thinking billed; 55-day abuse logs |
| Gemini 3 Pro Image | $0.134 (1K/2K), $0.24 (4K); batch half; no API free tier | same | 5 character + 6 object + 3 style refs | same; 16:9 up to 5504x3072 | same | same; highest price |
| Gemini 2.5 Flash Image | $0.039 | same | "best with up to 3 images" | 1344x768 for 16:9 | same | shutdown date October 2, 2026 already passed |
| OpenAI gpt-image-2.5 (sunburst/flare) | $30 per 1M image output tokens; per-image price UNVERIFIED; no Free row in rate tables | Customer "owns all Output"; C2PA + SynthID on API images | up to 16 input images; alpha mask (applied to first image, prompt-based); vendor notes recurring-character consistency can slip | Images API generations/edits, Responses tool; custom sizes e.g. 1536x864; no seed parameter documented | Tier 1: 100,000 TPM, 5 IPM; Tier 2: 20 IPM | ID verification gate; moderation blocks; up to 2 min latency; older models retire Oct 23 / Dec 1, 2026 |
| OpenAI gpt-image-2 | $0.005-$0.006 low, $0.041-$0.053 medium, $0.165-$0.211 high | same | same; always high-fidelity inputs | same | same table | no shutdown date found (UNVERIFIED) |
| Qwen-Image-Edit-2511 on L4 24 GB / 32 GB RAM | GPU time only; cost per image UNVERIFIED | Apache-2.0; no watermark mentioned | 1-3 input images (2509 card); pose via keypoint image; improved character consistency (2511 card) | diffusers `QwenImageEditPlusPipeline` / ComfyUI native; torch generator seed | none (self-hosted) | fp8 + text encoder = 29.9 GB vs 32 GB RAM; L4 speed UNVERIFIED; successor 2.1 is non-commercial |
| FLUX.2 klein 4B on RTX 4060 8 GB | free weights | Apache-2.0 | up to 4 references; unified edit + T2I | diffusers `Flux2KleinPipeline` / ComfyUI; torch generator seed | none | official VRAM claims conflict (8 vs 13 GB); 8 GB text encoder in 16 GB RAM; quality UNVERIFIED |
| FLUX.1 Kontext dev GGUF Q4 on 8 GB | free weights, but non-commercial model license | outputs "for any purpose (including for commercial purposes)" BUT model use for revenue-generating activity is not permitted without a BFL license | UNVERIFIED (gated card) | ComfyUI GGUF; seed | none | license trap; 6.8 GB of 8 GB VRAM |
| Z-Image-Turbo on 8 GB | free weights | Apache-2.0 | T2I only; edit variant "To be released" | diffusers `ZImagePipeline` / ComfyUI; seed | none | no reference-image path; needs LoRA for identity |
| SDXL + IP-Adapter on 8 GB | free weights | OpenRAIL++-M: licensor "claims no rights in the Output" | IP-Adapter Plus / Plus-Face image prompt; ControlNet | diffusers / ComfyUI; seed | none | fit, speed, identity all UNVERIFIED; FaceID needs InsightFace (license UNVERIFIED) |
| Qwen-Image-Edit Q2/Q3 or nunchaku on 8 GB / 16 GB | free weights | Apache-2.0 | as Qwen row | ComfyUI GGUF / nunchaku | none | weights 13.6-19.3 GB vs 16 GB RAM; highest OOM risk |

## Open questions / could not verify

1. Whether `gemini-2.5-flash-image` has actually stopped responding after its listed October 2, 2026 shutdown date.
2. Actual Gemini RPM / IPM / RPD numbers per tier for the image models (published only inside AI Studio, sign-in required).
3. Whether `generation_config.seed` gives reproducible images on Gemini image models.
4. Whether Gemini API images carry C2PA metadata in addition to SynthID.
5. Vertex AI output-ownership and indemnity terms (Google Cloud terms not opened).
6. Per-image dollar price of `gpt-image-2.5-sunburst` / `-flare` by quality and size (interactive calculator only).
7. Behaviour of `input_fidelity` on the `gpt-image-2.5` models; shutdown dates for `gpt-image-2` and the 2.5 models.
8. OpenAI usage policies text (page returned HTTP 403); the OpenAI Services Agreement PDF could be downloaded but not text-extracted, so the HTML version (en-GB URL) was used instead.
9. Any refusal behaviour of Gemini or GPT Image for this specific character (nothing was tested).
10. Speed of Qwen-Image-Edit on an NVIDIA L4 (no source found) and the resulting cost per image.
11. Whether third-party nunchaku builds of Qwen-Image-Edit-2511 are trustworthy; no official nunchaku-ai 2511 repo was found.
12. The 2511 recommended maximum number of input images.
13. FLUX.2 klein 4B real VRAM use on 8 GB (official sources say ~8 GB, 8.4 GB and ~13 GB) and its measured speed on an RTX 4060 with 16 GB RAM.
14. How Black Forest Labs applies the non-commercial model-use clause to a solo creator's commercial trailer, and the price of a commercial license for FLUX.1 Kontext [dev].
15. FLUX.1 Kontext [dev] model-card claims (gated; card not readable without login), text-encoder sizes, and 8 GB timings.
16. SDXL + IP-Adapter VRAM / speed on 8 GB and the InsightFace model license.
17. Comic / graphic-novel line-art quality and 5-view identity fidelity for every option (no benchmark found; none run).
18. One secondary page (sia.hackernoon.com article on FLUX.2 klein 4B) returned HTTP 403 and was not used.

## Sources

All read on 2026-10-05.

Google
- https://ai.google.dev/gemini-api/docs/image-generation - "Nano Banana image generation" (Gemini API docs)
- https://ai.google.dev/gemini-api/docs/pricing - "Gemini Developer API pricing"
- https://ai.google.dev/gemini-api/docs/rate-limits - "Rate limits" (Gemini API docs)
- https://ai.google.dev/gemini-api/docs/deprecations - "Gemini deprecations"
- https://ai.google.dev/gemini-api/terms - "Gemini API Additional Terms of Service" (effective March 23, 2026)
- https://ai.google.dev/gemini-api/docs/usage-policies - "Additional usage policies" (abuse monitoring, 55-day retention)
- https://ai.google.dev/gemini-api/docs/batch-api - "Batch API"
- https://ai.google.dev/api/interactions-api - "Interactions API" reference
- https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image - "Gemini 3.1 Flash Image" model page
- https://cloud.google.com/vertex-ai/generative-ai/pricing - "Vertex AI generative AI pricing"

OpenAI
- https://developers.openai.com/api/docs/guides/image-generation - "Image generation" guide
- https://developers.openai.com/api/docs/pricing - "Pricing"
- https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst - "GPT-Image-2.5 Sunburst" model page
- https://developers.openai.com/api/docs/models/gpt-image-2.5-flare - "GPT-Image-2.5 Flare" model page
- https://developers.openai.com/api/docs/models/gpt-image-2 - "GPT-Image-2" model page
- https://developers.openai.com/api/docs/models/gpt-image-1.5 and https://developers.openai.com/api/docs/models/gpt-image-1 - model pages (per-image price tables)
- https://developers.openai.com/api/reference/resources/images - "Images" API reference
- https://developers.openai.com/api/docs/deprecations - "Deprecations"
- https://developers.openai.com/api/docs/guides/your-data - "Data controls in the OpenAI platform"
- https://openai.com/en-GB/policies/services-agreement/ - "OpenAI Services Agreement" (updated 1 December 2025, effective January 1, 2026)
- https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images - Help Center, provenance signals (Content Credentials, SynthID) in OpenAI-generated content
- https://help.openai.com/en/articles/10910291-api-organization-verification - Help Center, API organization verification
- Not openable: https://openai.com/policies/usage-policies/ (HTTP 403); https://cdn.openai.com/osa/openai-services-agreement.pdf (downloaded, text not extractable)

Qwen / Alibaba
- https://huggingface.co/Qwen/Qwen-Image-Edit - model card
- https://huggingface.co/Qwen/Qwen-Image-Edit-2509 - model card
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511 - model card
- https://huggingface.co/Qwen/Qwen-Image-2.1 - model card
- https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE - "Qwen RESEARCH LICENSE AGREEMENT"
- https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I - model card (prompt rewriter)
- https://huggingface.co/Qwen - organisation model listing
- https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI - repackaged weights (file sizes)
- https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI - text encoder / VAE (file sizes)
- https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF - GGUF quantizations
- https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF - GGUF quantizations
- https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning - Lightning LoRAs
- https://huggingface.co/lightx2v/Qwen-Image-Lightning - Lightning LoRAs
- https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509 - nunchaku 4-bit models
- https://huggingface.co/QuantFunc/Nunchaku-Qwen-Image-EDIT-2511 - third-party nunchaku build (existence only)
- https://github.com/nunchaku-ai/nunchaku - nunchaku README
- https://nunchaku.tech/docs/nunchaku/usage/qwen-image.html - nunchaku docs, Qwen-Image usage
- https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage - diffusers QwenImage pipelines
- https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511 - ComfyUI docs, Qwen-Image-Edit-2511

Black Forest Labs / FLUX
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B - model card
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8 - fp8 weights
- https://huggingface.co/black-forest-labs/FLUX.2-klein-9B - model repo (license tag)
- https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF - GGUF quantizations
- https://github.com/black-forest-labs/flux2 - FLUX.2 README
- https://github.com/black-forest-labs/flux2/blob/main/model_licenses/LICENSE-FLUX-NON-COMMERICAL - FLUX Non-Commercial License v2.1
- https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev - FLUX.1 [dev] Non-Commercial License v1.1.1
- https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev - model repo (license tag only; card gated)
- https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF - GGUF quantizations
- https://huggingface.co/nunchaku-ai/nunchaku-flux.1-kontext-dev - nunchaku build (license tag)
- https://huggingface.co/black-forest-labs/FLUX.1-schnell - model repo (license tag)
- https://huggingface.co/city96/FLUX.1-schnell-gguf - GGUF quantizations
- https://docs.bfl.ai/flux_2/flux2_overview - BFL docs, FLUX.2 overview
- https://bfl.ai/blog/flux2-klein-on-device - BFL blog, June 4, 2026
- https://bfl.ai/licensing - BFL licensing page
- https://docs.comfy.org/tutorials/flux/flux-2-klein - ComfyUI docs, FLUX.2 klein
- https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2 - diffusers Flux2 pipelines

Other local models
- https://huggingface.co/Tongyi-MAI/Z-Image-Turbo - model card
- https://huggingface.co/Tongyi-MAI/Z-Image - model card
- https://huggingface.co/Tongyi-MAI - organisation model listing
- https://huggingface.co/Comfy-Org/z_image_turbo - repackaged weights (file sizes)
- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 - model card
- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md - CreativeML Open RAIL++-M license
- https://huggingface.co/h94/IP-Adapter - model card
- https://huggingface.co/h94/IP-Adapter-FaceID - model card
- https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl - diffusers SDXL guide

Secondary / anecdotal
- https://www.thundercompute.com/blog/qwen-image-edit-comfyui - Thunder Compute blog, June 9, 2026
- https://lilting.ch/en/articles/qwen-image-edit-2511-local-specs - lilting.ch article, January 14, 2026 (figures conflict with HF file sizes)
- https://forums.developer.nvidia.com/t/qwen-image-2-1-on-dgx-spark-54-s-per-1024-image-40-steps-31-6-gb-and-a-fix-for-the-pink-vertical-line/384774 - NVIDIA developer forum post, September 30, 2026
- https://localaimaster.com/vram/best-flux-model-8gb-vram - localaimaster.com, published July 7, 2026, updated August 2026
- https://aistudynow.com/comfyui-qwen-image-edit-low-vram-workflow-gguf-vs-fp8/ - opened, contained no measured numbers, not used for any claim
- Not openable: https://sia.hackernoon.com/the-8gb-vram-image-model-that-feels-instant-meet-flux2-klein-4b (HTTP 403)
