# Explanation

1. Explain motivation and how each filter works.

# Ideas for Robustness

1. relative color detection
2. possibly green cloth


# Possible Code Upgrades


1. In `def segment_table(img):`, we can include this in future versions. It's a werid concept but supposedly clears up noise if needed.

```py
# Clean up: close small holes, then open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
```

# Questions

1. Are we also supposed to identify balls inside pockets when that happens??


# Problems

1. How do we detect balls where the entire circle isnt visible (Like when a hand or edge of the table covers part of it), and simultaneously NOT the pockets?? 
2. What better ways can we detect balls with low color difference in grayscale?
