import seaborn as sns
import matplotlib.pyplot as plt
# Draw Plot
plt.figure(figsize=(8, 4), dpi=80)
sns.kdeplot([1,2,3,4], shade=True, color="g", label="Proba", alpha=.7, bw_adjust=0.01)
# Decoration
plt.title('Density Plot of Probability', fontsize=22)
plt.legend()
plt.show()