# Datasets for Learning Path Analysis

## Sequence Grading

**Possible Evaluation Fields on Learning Resource Level:**
     

|     | Possible Grading Metrics for a Sequence  |
| --- | ---------------------------------------- |
|     | Avg # Correct Learning Resources         | 
|     | Avg # Hints per Learning Resource        |
|     | Avg # Attempts per Learning Resource     |
|     | Avg Score per Learning Resource          |
|     | Overall Grade for Sequence(if available) |
---

## Path Similarity

- Use a sequence distance approach to calculate the similarity between all paths(i.e. user combinations per topic)

---

## Group Paths Together: Clustering of distance matrix

| _     | Path1 | Path2 | Path3 |
| ----- | ----- | ----- | ----- |
| Path1 | sim   | sim   | sim   |
| Path2 | sim   | sim   | sim   |
| Path3 | sim   | sim   | sim   |

---

## Evaluation Algorithm

1. Calculate an aggregate grading for a sequence  

2. Cluster sequences via sequence similarity matrix  


