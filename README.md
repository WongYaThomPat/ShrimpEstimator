# ShrimpEstimator
- Run shrimp_connect.py to see the visuals.
- Run validate.py to get the performance metrics.

# How it works
1. A custom CNN (basically the encoder half of U-Net) get the keypoints (head, body, tail).
2. Using KNN and some geometrical constraints to reconstruct those keypoints into Shrimps.
3. Find UnitPerPixel via visual reference. In this case it's the net.
4. Do Quadratic Interpolation and Distance Transform on each Shrimp to get estimated Length and Diameter respectively.
5. Multiply those with UnitPerPixel to get the final estimated values in real world units.
