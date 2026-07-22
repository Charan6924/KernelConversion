import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/home/cxv166/PhantomTesting/Code/training_output_kernel256/training_metrics_20260521_084853.csv')

print(df.head())

plt.plot(df['train_ft_loss'])
plt.title('f_loss')
plt.savefig('ft_loss.png')
plt.clf()


plt.plot(df['val_recon_loss'])
plt.title('val recon loss')
plt.savefig('recon_loss')
plt.clf()


plt.plot(df['val_total_loss'])
plt.title('val total loss')
plt.savefig('total_loss')
plt.clf()

plt.plot(df['train_grad_norm'])
plt.title('train grad norm')
plt.savefig('grad_norm')
plt.clf()
