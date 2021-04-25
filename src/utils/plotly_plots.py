import functools


def accuracy_plot(func, title):
  opts = dict(xlabel='Steps',
              ylabel='Accuracy',
              title=title,
              ytickmin=0,
              ytickmax=1.05)
  return functools.partial(func, update='append', win=title, opts=opts)
