# Evaluate both models on the plaid mask sequence
cnn_model.eval()
cnn_plaid_curve = []

with torch.no_grad():
    for mc in mask_contrasts:
        mask = create_grating(theta=mask_angle, contrast=mc)
        plaid = base_stim + mask
        
        plaid_tensor = torch.tensor(plaid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred = cnn_model(plaid_tensor).item()
        cnn_plaid_curve.append(pred)

# Visualizing the Head-to-Head Comparison
plt.figure(figsize=(8, 5))
plt.plot(mask_contrasts, dn_plaid_curve, color="navy", lw=2.5, label="Divisive Normalization (Biophysical)")
plt.plot(mask_contrasts, cnn_plaid_curve, color="crimson", lw=2.5, linestyle="-.", label="Backprop CNN")
plt.plot(mask_contrasts, ln_plaid_curve, color="orange", lw=1.5, linestyle="--", label="Standard LN Model")

plt.title("Generalization Benchmark: Cross-Orientation Suppression", fontsize=13)
plt.xlabel("Orthogonal Mask Contrast", fontsize=11)
plt.ylabel("Predicted Response to Preferred Stimulus (dF/F)", fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
