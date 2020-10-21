import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# render_iter = [ 104, 259, 362, 465, 568, 671, 774, 877, 980, 1083]
# render_time = [ 1.24543, 3.04093, 4.31031,  5.61925, 6.78034,  8.0182, 9.21798, 10.4139, 11.7103, 12.8612]
# render_time = [ render_time[i] / render_iter[i] for i in range(0, len(render_iter))]
# plt.plot(render_iter, render_time, label="base")

# plt.title("Average time per iteration")
# plt.xlabel("First # of iterations")
# plt.ylabel("Average time (s)")
# plt.show()

# denoise_time = [0.004441, 0.0049572, 0.0065664, 0.0072701, 0.0082553, 0.0106892, 0.0108782, 0.0116914]
# x = [i for i in range(0, len(denoise_time))]
# plt.plot(x, denoise_time)

# plt.title("Denoise time")
# plt.xlabel("# of denoise levels")
# plt.ylabel("Denoise time (s)")
# plt.show()


resolution = [400, 500, 600, 700, 800, 900, 1000]
resolution_time = [ 0.0023007,  0.0033596, 0.0046226, 0.0060183, 0.0082553, 0.0107577,  0.0126112]
plt.plot(resolution, resolution_time)

plt.title("Denoise time for 5 denoise levels")
plt.xlabel("One side resolution")
plt.ylabel("Denoise time (s)")
plt.show()



