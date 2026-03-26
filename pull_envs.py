import orbax.checkpoint as ocp
import jax.numpy as np
import matplotlib.pyplot as plt
import os

checkpoint_path = "/cs/student/project_msc/2025/csml/rhautier/jaxued_results/initial_accel_run/0/sampler/default"

checkpointer = ocp.StandardCheckpointer()

restored = checkpointer.restore(checkpoint_path)
print('keys found', restored.keys())


levels = restored['sampler']['levels']
scores = restored['sampler']['scores']

fig, axes = plt.subplots(2,5, figsize =(15,6))

for i, ax in enumerate (axes.flat):

    ax.imshow(levels[i], cmpa='gray')
    ax.set_title(f"Buffer Level {i}, regret {scores[i]}")

    ax.axis('off')



