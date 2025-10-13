set multiplot layout 2,1
plot x*x - 3*x + 5, "plot_data.txt" with lines
plot "cost_plot_data.txt" with lines