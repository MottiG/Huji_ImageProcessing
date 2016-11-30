from sol2 import *

papo = np.random.random(20)
# papo = np.copy(papo).reshape(20,1)


r1 = DFT(papo)
r2 = np.fft.fft(papo)
# r1= DFT(papo)
# r3=DFT(papo2)
# r2 = (np.fft.fft2(papo2))
print(r1.dtype,r2.dtype)
print(np.allclose(r1,r2))