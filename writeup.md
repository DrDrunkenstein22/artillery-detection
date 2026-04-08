# Military Object Detection: A Beginner-Friendly Writeup
### Comparing AI Models for Real-Time Artillery & Vehicle Detection

---

## What Are We Actually Building?

Imagine a drone or a surveillance camera on a battlefield. It captures video constantly. A human analyst watching that footage would get exhausted, miss things, and react slowly. Now imagine instead that a computer watches that same footage and instantly highlights every tank, artillery gun, or military truck it sees — in real time.

That's exactly what this project does.

We trained and compared **three different AI models** to detect military vehicles and weapons in images. The goal is to find out which model is the best fit for real-world military use — specifically whether it's fast enough, accurate enough, or both.

---

## The Problem We're Solving

Artillery and armored vehicles are hard to spot, especially from aerial images or at a distance. They can be camouflaged, partially hidden, or moving fast. Manual detection is slow and error-prone.

An AI system that can automatically detect and classify these objects could:
- Give soldiers faster situational awareness
- Reduce the workload on intelligence analysts
- Work 24/7 without fatigue
- Process dozens of video feeds simultaneously

---

## What Can Our System Detect?

We trained the models to recognize **8 types of military objects**:

| Class | What It Is |
|---|---|
| **Artillery** | Towed or self-propelled guns — the primary target |
| **Tank** | Main battle tanks (e.g. T-72, Abrams) |
| **APC** | Armored Personnel Carriers — troop transport vehicles |
| **Military Truck** | Logistics and transport trucks |
| **Rocket Artillery** | Multi-launch rocket systems (MLRS) like Grad or HIMARS |
| **IFV** | Infantry Fighting Vehicles — armored but lighter than tanks |
| **Military Aircraft** | Fighter jets, military helicopters |
| **Other Military** | Anything military that doesn't fit the above |

Artillery gets extra attention in training — it appears twice as often in our training data compared to other classes — because it's the most critical object to detect correctly.

---

## The Data: Where Did the Images Come From?

We used two publicly available datasets:

- **Kaggle** (`military-assets-dataset`) — ~26,000 labeled images of military objects
- **Roboflow** (`mil-det`) — ~9,800 labeled images focused on military objects

The problem: both datasets used **different names for the same things**. One called it "panzer", the other "tank". One said "mlrs", the other "rocket-artillery". Our code automatically translates all these different names into our 8 standard categories.

After cleaning and merging, we ended up with **~20,300 usable images**, split like this:
- **80% for training** (~16,200 images) — what the model learns from
- **10% for validation** (~2,000 images) — checked during training to avoid overfitting
- **10% for testing** (~2,000 images) — the final exam the model has never seen

---

## The Three Models: Why These Three?

We picked three models that represent **three different philosophies** of how to do object detection. Comparing them tells us which approach works best for our specific problem.

---

### Model 1: YOLOv11m 

**YOLO** stands for "You Only Look Once." The name says it all.

Most older AI systems looked at an image multiple times, in different ways, before deciding what was in it. YOLO changed everything by looking at the entire image just once and spitting out all detections instantly.

**How it works (simply):**
> Divide the image into a grid. Each grid cell asks itself: "Is there an object here? If yes, what is it and where exactly?" All grid cells answer at the same time, in parallel. Done.

**Why we chose it:**
- Extremely fast — can run on live video in real time
- Well-proven in military and drone applications
- YOLOv11 is the latest generation, more accurate than older versions
- The "m" means medium-sized — a balance between speed and accuracy

**Best for:** Real-time video feeds, drone surveillance, situations where you need an answer *now*.

---

### Model 2: RT-DETR-L 

**RT-DETR** stands for Real-Time DEtection TRansformer. It uses a technology called a **Transformer**, which originally became famous in AI language models (the same technology behind ChatGPT).

The key idea with Transformers is **attention** — the model doesn't just look at one part of the image at a time. It looks at the whole image and figures out which parts are most relevant to each other.

**How it works (simply):**
> Instead of a grid, the model reads the whole image like reading a sentence — understanding context. A tank partially hidden behind a tree? The model understands that the visible parts "relate to" each other and predicts there's a full tank there. It finds objects directly, without needing any pre-set grid or sliding window.

**Why we chose it:**
- More accurate than YOLO on complex or cluttered scenes
- Better at detecting partially hidden objects
- Still fast enough for near-real-time use (the "RT" = Real-Time)
- The "L" means large — more powerful but slightly slower than YOLO

**Best for:** Situations where accuracy matters more than raw speed — intelligence analysis, satellite image review.

---

### Model 3: Faster R-CNN R50 

**Faster R-CNN** is an older, well-established architecture from 2015. It's a **two-stage detector**, meaning it works in two steps:

**How it works (simply):**
> **Step 1 — Find candidates:** Scan the image and mark every region that *might* contain an object. These are called "region proposals."
> **Step 2 — Classify:** Look at each candidate region carefully and decide: what exactly is this, and where precisely is it?

This two-step process makes it slower but very thorough. It doesn't miss much.

The "R50" means it uses **ResNet-50** as its backbone — a deep neural network that's very good at understanding visual features, trained on millions of everyday images before being fine-tuned on our military data.

**Why we chose it:**
- Considered the gold standard in accuracy for many years
- Represents the "traditional" approach — great baseline for comparison
- Well-understood, used in many real research papers
- Slower than YOLO/RT-DETR but often more precise

**Best for:** Offline analysis where you have time to process images carefully — reviewing satellite imagery, post-mission analysis.

---

## How We Trained Them

Training an AI model means showing it thousands of labeled images and letting it slowly adjust itself until it gets good at the task. Think of it like flashcards — show the model an image, ask "what's in this?", compare its guess to the correct answer, and nudge it in the right direction. Repeat 50 times over the entire dataset.

We trained all three models on **cloud GPUs** (NVIDIA T4) using a platform called **Modal**, which lets us rent GPU time without owning expensive hardware. All three models trained simultaneously in parallel to save time.

**Training settings:**
- YOLOv11: 50 rounds (epochs), images resized to 640×640 pixels
- RT-DETR: 50 rounds, same image size, smaller batch size (RT-DETR is heavier)
- Faster R-CNN: 25 rounds, larger images at 800px (needs more detail for its two-stage process)

---

## How We Measure "Good"

We use two main metrics:

**mAP50 (mean Average Precision at 50% overlap)**
> For a detection to count as correct, the box the model draws around an object must overlap at least 50% with the actual labeled box. mAP50 averages this score across all 8 classes. Higher = better. 1.0 is perfect.

**FPS (Frames Per Second)**
> How many images the model can process every second. For real-time video (30 fps), the model needs to keep up. Higher = faster.

The trade-off: accuracy vs speed. A model that's very accurate but slow is useless on a live drone feed. A model that's fast but misses half the targets is dangerous.

---

## Real-World Military Use Cases

**1. Drone and UAV Surveillance**
Autonomous drones can fly over contested territory and automatically flag any detected military vehicles, sending alerts back to operators without requiring a human to watch every second of footage.

**2. Border Monitoring**
Fixed cameras at borders or checkpoints can continuously scan for military vehicle movements, flagging unauthorized crossings in real time.

**3. Satellite Imagery Analysis**
Intelligence agencies receive thousands of satellite images daily. AI can pre-screen these images and flag only the ones that contain military activity, saving analyst time.

**4. Artillery Fire Control Support**
Forward observers could use AI on a tablet or ruggedized device to rapidly identify and classify enemy artillery positions, feeding coordinates to fire control systems faster than manual methods.

**5. Convoy Protection**
 Vehicles in a convoy could use onboard cameras with AI to detect ambush threats — spotting hidden artillery or IFVs before they engage.
 
---

## Why This Comparison Matters

There's no single "best" model — it depends on the situation:

| Situation | Best Choice | Why |
|---|---|---|
| Live drone video | YOLOv11 | Fastest, runs in real time |
| Analyzing satellite imagery | Faster R-CNN | Most thorough, accuracy matters |
| Near-real-time with good accuracy | RT-DETR | Best of both worlds |
| Low-power edge device | YOLOv11 | Smallest footprint |
| Partially obscured vehicles | RT-DETR | Transformer attention handles occlusion |

By running all three on the same dataset with the same classes and evaluation method, we get an honest, apples-to-apples comparison that tells us exactly where each model wins and where it falls short.

---

## The Bottom Line

We built a system that:
1. **Collects and cleans** real military image data from two sources
2. **Trains three state-of-the-art AI models** to detect 8 types of military objects
3. **Compares them fairly** on accuracy and speed
4. **Gives actionable results** — which model to deploy, and when

The end result is a research-backed answer to a real operational question. 
