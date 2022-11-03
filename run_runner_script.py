import os
import subprocess


def combs(a):
    if len(a) == 0:
        return [[]]
    cs = []
    for c in combs(a[1:]):
        cs += [c, c + [a[0]]]
    return cs


test = combs([0, 1, 2, 3, 4, 5, 6, 7, 8])
runner_cwd = os.path.join(os.getcwd(), 'runner.py')

# for ii in test:
#     if len(ii) == 7:
#         in_index = sorted(ii)
#         print(in_index)
#         subprocess.call(['python', str(runner_cwd), '--img_indexes', str(in_index), '--n_view_renderings', str(len(in_index))])
#         os.rename('output', 'output_' + str(in_index))

for ii in range(0, 10):
   subprocess.call(['python', str(runner_cwd)])
   os.rename('output', 'rerun_output_' + str(ii))
