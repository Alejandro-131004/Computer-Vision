# Computer Vision Project — Task 1 Summary

Source: :contentReference[oaicite:0]{index=0}

## Overview

Task 1 is an **image-processing/computer-vision task** focused on **8-ball pool table analysis**. The system must take images of a pool table and produce structured outputs about the balls and the table layout.

## Objective

Given an image of an **8-ball pool table**, the solution must determine:

1. **Total number of balls on the table**
2. **Position of each ball in the image**
   - specifically via **bounding boxes**
3. **Number of each ball**
4. **Top-view of the table**

## Input

- An **image of an 8-ball pool table**
- In the final pipeline description, the input is also described as a **list of images**

## Required Outputs

### Per-image analytical output
For each input image, the system must produce:

- The **total count of balls**
- The **bounding box** of each detected ball
- The **ball number** for each detected ball
- A **top-view transformed version** of the table

### Batch pipeline outputs
The final implementation must operate as a pipeline with:

- **Input:** list of images
- **Output:** list of results
- **Additional output:** a directory containing the generated top-view images, named as:
  - `<image1_name>.jpg`
  - `<image2_name>.jpg`
  - `...`
  - `<imageN_name>.jpg`

## Ball Number Identification Rule

Ball numbers may be inferred **based on colour**.

Specific instruction:

- The **cue ball (white ball)** must be considered **ball number 0**

## Dataset Information

The task references a public dataset:

- **Roboflow dataset:** `8-ball-pool-l530o`

Also stated for the assignment dataset setup:

- **50 images** are randomly chosen from a public dataset
- Evaluation will be performed on **10 undisclosed images**

## Expected Technical Approach

The deliverable is framed as an **image-processing pipeline**.

### Allowed tools/libraries
The solution should use:

- **OpenCV**
- Other **common libraries**, explicitly including:
  - `numpy`
  - `matplotlib`

This strongly indicates the work is expected to be based on **classical computer vision / image processing**, not a heavy custom framework stack.

## Deliverables

You must submit:

1. **Short report**
   - maximum **2 pages**
   - should present:
     - the **methodology**
     - **some results**

2. **Python script**
   - **only one file**

## Deadline

- **April 17**
- **23:59 AoE**

## Grading and Weight

- **Task 1 accounts for 40% of the overall project grade**

### Evaluation criteria
The following elements will be considered:

- **Methodology**
- **Report**
- **Quality of the results**

## Important Restrictions and Remarks

These constraints are stated explicitly and should be treated as mandatory:

### 1. Follow the JSON format strictly
- You must **follow strictly the JSON structure** for both:
  - **input files**
  - **output files**

### 2. AI usage is allowed only with acknowledgment
- It is acceptable to use **AI tools** while developing the work
- It is **not acceptable** to use AI tools **without acknowledging it**

### 3. All group members are accountable
- **All members of the group** are expected to:
  - understand the **methodology**
  - understand the **submitted code**

## What the task is really asking for

At a practical level, Task 1 requires a system that can:

1. Detect the pool table and/or rectify perspective
2. Detect all visible balls
3. Localize them with bounding boxes
4. Infer ball identities from colour
5. Generate a normalized **top-view table image**
6. Return all outputs in the required JSON format
7. Save top-view images to an output directory

## Non-negotiable compliance checklist

- [ ] Input handled as image / list of images
- [ ] Detect all balls present on the table
- [ ] Count total number of balls
- [ ] Produce bounding boxes for each ball
- [ ] Infer ball number from colour
- [ ] Treat cue ball as number `0`
- [ ] Generate top-view of the table
- [ ] Return list of results
- [ ] Save top-view images as `<image_name>.jpg`
- [ ] Use a single Python file
- [ ] Use OpenCV + common libraries only
- [ ] Write a report of at most 2 pages
- [ ] Respect the exact JSON input/output structure
- [ ] Acknowledge any AI assistance used
- [ ] Ensure every group member understands the methodology and code

## Concise interpretation for implementation planning

This is a **single-file Python computer-vision pipeline** for **8-ball pool analysis**. It must process a set of images, detect and identify balls, estimate their positions, create a top-down view of the table, and export both structured JSON results and saved top-view images, while respecting strict formatting and submission constraints.