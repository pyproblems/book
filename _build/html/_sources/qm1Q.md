# Problem 1

**Time evolution before and after measurement**

An arbitrary wavefunction is confined in an infinite square well of width L. Some time later a measurement is made of the particle's position. Assume that the uncertainty on this position measurement is Gaussian in form with a standard deviation of L/10.


1. Produce an animation showing the time evolution of this wavefunction before and after the position measurement.


```{note}
You can use the following packages to produce an animation which you can playback in the Python notebook:

`from IPython.display import HTML`
`from matplotlib import animation`

When produce your animation use the following commands to display in the notebook:

`anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes)
HTML(anim.to_jshtml())`
```