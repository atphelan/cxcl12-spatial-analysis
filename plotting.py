#%%
from typing import List
import collections
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import plotly.io as pio
# print('PIO renderers: ', pio.renderers)
pio.renderers.default = "browser"

from utils import Folder

label_lookup = {
    'Mean distance to 5 nearest neighbours': "d5"
}

#%%
def plotly_line(df, x, y, title=None, linewidth=6, fillcolour='blue'):
	layout = go.Layout(plot_bgcolor='white')
	if x == 'index':
		df['index'] = df.index

	fig = go.Figure(data=go.Scatter(x=df[x], y=df[y], mode='lines',
		line = dict(color=fillcolour, dash='solid', width=linewidth),
		name = 'Observed dispersion',
	), layout=layout)

	if y == 'Ripley L':
		fig.add_trace(go.Scatter(x=df[x], y=df[x], mode='lines',
			line = dict(color='black', dash='dash'),
			name = 'Uniform distribution',
		))

	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=x, font=dict(size=24)),
		yaxis_title=dict(text=y, font=dict(size=24)),
		title=dict(text=title, font=dict(size=12)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=60, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="left",
			x=0.05,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		nticks=10,
		# range=[0.0, 12.01],
		# dtick=20,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=1,
		# range=[0.0, 9.5],
	)
	return fig


#%%
def plotly_scatter(df, x, y, c, colourscheme='Viridis', trendline=False, title=None, markersize=6, fillcolour='blue'):
	layout = go.Layout(plot_bgcolor='white')
	if x == 'index':
		df['index'] = df.index
	if c is None:
		fig = go.Figure(data=go.Scatter(x=df[x], y=df[y], mode='markers',
			marker=dict(
				size=markersize,
				# line=dict(width=20/np.sqrt(len(df)), color='darkblue'),
				color=fillcolour,
			),
		), layout=layout)
	else:
		fig = go.Figure(data=go.Scatter(x=df[x], y=df[y], mode='markers',
			marker=dict(
				size=markersize,
				line=dict(width=20/np.sqrt(len(df)), color='darkblue'),
				color=df[c],
				colorscale=colourscheme,
				showscale=True,
				colorbar={'title':label_lookup.get(c, None), 'nticks': 10, 'thickness': 18}
			),
		), layout=layout)
	# if trendline:
	# 	fig.add_trace(

	# 	)
	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=x, font=dict(size=24)),
		yaxis_title=dict(text=y, font=dict(size=24)),
		title=dict(text=title, font=dict(size=12)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=60, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="left",
			x=0.05,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		nticks=10,
		# range=[0.0, 12.01],
		# dtick=20,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=1,
		# range=[0.0, 9.5],
	)
	return fig

#%%

def plotly_2lines(df1, x1, y1, df2, x2, y2, colours=('red', 'blue')):
	layout = go.Layout(
		plot_bgcolor='white',
		yaxis2=dict(
			overlaying='y',
			side='right',
			showgrid=False,
			zeroline=False,
			linewidth=1.5,
			linecolor='black',
			mirror=True,
		)
	)

	fig = go.Figure(layout=layout)

	# First line (left y-axis)
	fig.add_trace(
		go.Scatter(
			x=df1[x1],
			y=df1[y1],
			mode='lines',
			name=y1,
			line=dict(color=colours[0], width=3, dash='dot'),
		)
	)

	# Second line (right y-axis)
	fig.add_trace(
		go.Scatter(
			x=df2[x2],
			y=df2[y2],
			mode='lines',
			name=y2,
			# yaxis='y2',
			line=dict(color=colours[1], width=3, dash='dash'),
		)
	)

	fig.update_layout(
		# xaxis_title=dict(text=label_lookup.get(x1, None), font=dict(size=24)),
		yaxis_title=dict(text=y1, font=dict(size=24)),
		yaxis2_title=dict(text=y2, font=dict(size=24)),
		autosize=False,
		height=400,
		width=600,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="right",
			x=0.95,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)

	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		nticks=13,
		range=[1.0, 12.01]
	)

	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# range=[0.0, 45.0]
		scaleanchor="x",
		scaleratio=1,
	)

	return fig

cs = ['cornflowerblue', 'darkgreen', 'indianred']
colours_lookup = {1:cs[0], 2:cs[2], 3:cs[1], 4:cs[0], 5:cs[2], 6:cs[2], 7:cs[1], 8:cs[0]}
colours_list = ['cornflowerblue', 'crimson', 'darkgreen', 'deepskyblue', 'darkred', 'indianred', 'green', 'mediumslateblue']
slide_colours = {f'Slide {n}': colours_lookup[n] for n in range(1, 9)}


#%%
def plotly_boxplot(data_dict, y_name, legend=False):
	'''
	Specify category name for x axis as key to df_dict, with array of y as corresponding value
	'''
	fig = go.Figure(layout=go.Layout(plot_bgcolor='white'))
	for k, df in data_dict.items():
		fig.add_trace(go.Box(
			y=df[y_name],
			quartilemethod='inclusive',
			name=k,
			# marker_color=slide_colours[k[:7]]
		))

	if legend:
		full_name_lookup = {k[:7] : k for k in data_dict.keys()}
	else:
		fig.update_layout(showlegend=False)

	fig.update_layout(
		# xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		# yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis_title=dict(text=y_name, font=dict(size=24)),
		# yaxis_title=dict(text=r'$\text{Local density (cells / 100 }\mu m^2\text{)}$', font=dict(size=24)),
		# autosize=False,
		height=800,
		width=800,
		# margin=dict(l=0, r=0, t=60, b=0),
		# hovermode='closest',
		font=dict(size=26),
		# boxmode='group',
		# showlegend=False,
	)

	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# nticks=10,
		# range=[0.0, 12.01],
		# dtick=20,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		# nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=1,
		# range=[0.0, 9.5],
	)

	if legend:
		fig.update_layout(
			xaxis=dict(
				tickmode="array",
				tickvals=list(full_name_lookup.values()),
				ticktext=list(full_name_lookup.keys())
			)
		)

	return fig


#%%
def plotly_cdf_curve(df, x, cmap, title=None):
	df = df.sort_values('Local density (cells / 100 um^2)').reset_index()
	renaming = {'Local density (cells / 100 um^2)': r'$\text{Local density (cells / 100 } \mu m^2\text{)}$'}
	df.rename(columns=renaming, inplace=True)
	if x in renaming.keys():
		x = renaming[x]
	df['CDF'] = df.index/len(df)
	max_x = 0.18
	if df[x].max() < max_x:
		df = pd.concat([df, pd.DataFrame({x: np.linspace(df[x].max(), max_x, 10), 'CDF': np.ones(10)})])

	return plotly_line(df, x, 'CDF', title=title, linewidth=4, fillcolour=slide_colours[title[:7]])

#%%
from clustering_analysis import ripley_k_l, compute_area_mask, compute_area
from scipy.spatial import cKDTree

def plotly_rkl_curve(df, x, y, title=None):
	coords = df[[x, y]].values
	tree = cKDTree(coords)
	area = compute_area_mask(coords, buffer_radius=60) # Larger buffer will exaggerate tissue area
	radii = np.logspace(0, 2, 30)
	radii = np.append(radii, np.linspace(100, 1000, 90))
	K, L = ripley_k_l(coords, radii, area, tree=tree)

	L_minus_r = L - radii


	kldf = pd.DataFrame({'Distance from a cell': radii, 'Ripley L': L, 'Ripley L - distance': L_minus_r})
	kldf.dropna(inplace=True)
	return plotly_line(kldf, 'Distance from a cell', 'Ripley L', title=title, linewidth=4, fillcolour=slide_colours[title[:7]])

#%%
def demo_plot(df, x, y, c, lookup_df, colourscheme='Viridis'):
	'''
	Plot scatterpoints on a graph for x and y, color points by c and show how close they are to the ground truth
	'''
	pass
	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(data=go.Scatter(x=df[x], y=df[y], mode='markers',
		marker=dict(
			size=6,
			line=dict(width=0.5, color='darkblue'),
			color=df[c],
			colorscale=colourscheme,
			showscale=True,
			colorbar={'title':c, 'nticks': 5, 'thickness': 14}
		),
		name='Simulated data',
	), layout=layout)

	groundtruth_df = lookup_df.loc[np.logical_and(lookup_df['BrdU time'].isin(df['BrdU time'].unique()), np.logical_and(
		np.isclose(lookup_df['k1'], df['k1'].mean(), atol=1E-7), np.isclose(lookup_df['k2'], df['k2'].mean(), atol=1E-7)))].copy()
	neighbourhood_df = lookup_df.loc[np.logical_and(lookup_df['BrdU time'].isin(df['BrdU time'].unique()), np.logical_and(
		np.isclose(lookup_df['k1'], df['k1'].mean(), rtol=1E-1), np.isclose(lookup_df['k2'], df['k2'].mean(), rtol=10)))].copy()

	fig.add_traces(data=[go.Scatter(x=groundtruth_df[x], y=groundtruth_df[y], mode='markers',
		marker=dict(
			size=12,
			line=dict(width=0.5, color='darkblue'),
			color=groundtruth_df['BrdU time'],
			symbol='x',
			showscale=False,
			colorscale='Greys',
			# colorbar={'title':'BrdU time', 'nticks': 2, 'thickness': 10,}
		),
		name='True mean'),
		go.Scatter(x=neighbourhood_df[x], y=neighbourhood_df[y], mode='markers',
		marker=dict(
			size=2,
			color='black',
			symbol='circle-dot',
		),
		name='Nearby true means'
	)])

	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=x, font=dict(size=24)),
		yaxis_title=dict(text=y, font=dict(size=24)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="bottom",
			y=0.05,
			xanchor="right",
			x=0.95,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		nticks=10,
		# range=[0.0, 12.01],
		# dtick=20,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=1,
		# range=[0.0, 45.0],
		scaleanchor="x",
		scaleratio=1,
	)
	return fig


#%%
def plotly_bar(categories: List[str], heights: List[float], colours: List[str]):
	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(data=[go.Bar(
		x=categories,
		y=heights,
		marker_color=colours,
		marker_line_width=1.5,
		marker_line_color='black')
	], layout=layout)
	fig.update_layout({'bargap': 0})
	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=None, font=dict(size=24)),
		yaxis_title=dict(text=None, font=dict(size=24)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="left",
			x=0.05,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=20,
		# range=[0.0, 9.5],
	)
	return fig


#%%
# Source - https://stackoverflow.com/a/51439172
# Posted by Sandu Ursu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-24, License - CC BY-SA 4.0

def qqplot(df, xvar):
    logxvar = np.log(df[xvar])
    qq = stats.probplot(logxvar, dist='lognorm', sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    fig = go.Figure()
    fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
    fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
    fig.layout.update(showlegend=False)
    return fig

def histplot(df, x, ref_mu, ref_sigma, bins=10, title=None):
	ax = df[x].hist(bins=bins)
	xs = np.linspace(df[x].min(), df[x].max(), 100)
	ax.plot(xs, stats.norm.pdf(xs, loc=ref_mu, scale=ref_sigma) * len(df)*(xs[-1]-xs[0])/bins)
	ax.set_xlabel(x)
	ax.set_ylabel('Frequency')
	ax.set_title(title)
	return ax

def hole_plots_and_stats():
	df['Mouse number'] = df['name'].apply(lambda s: ('2' if 'M2' in s else '4' if 'M4' in s else '5' if 'M5' in s else '2'))
	df['Bone type'] = df['name'].apply(lambda s: ('Hip' if 'Hip' in s else 'Femur' if 'Fem' in s else 'ThV'))
	# Some MANOVA for bone type + mouse number affecting any CXCL12/hole density/count?

#%%
if __name__ == '__main__':
	# df = pd.read_csv('data/CS_CX_HLF_12h_Femur_M4_annotations_measure_final_analysed.csv')
	data_dict = {}
	for file in Folder('data').files:
		if ('analysed' in file.name and not 'density' in file.name) and ('Slide' in file.name and 'duplicates' in file.name):
			df = file.read()
			fname = file.name.split('_duplicates')[0]
			data_dict.update({fname: df.copy()})

			# fig = plotly_scatter(df, 'X', 'Y', 'Mean distance to 5 nearest neighbours', title=file.name.split('_analysed')[0])
			# fig.write_image('images/' + file.name[:-4] + 'scatter.png')
			
			# fig = plotly_scatter(df, 'index', 'Mean distance to 5 nearest neighbours', 'Mean distance to 5 nearest neighbours')
			# fig.show()
			
			# fig = qqplot(df, 'Mean distance to 5 nearest neighbours')
			# fig.show()
			
			# ax = histplot(
			#     df,
			#     'Mean distance to 5 nearest neighbours',
			#     ref_mu=df['Mean distance to 5 nearest neighbours'].mean(),
			#     ref_sigma=df['Mean distance to 5 nearest neighbours'].std(),
			#     bins=30,
			# 	title=file.name.split('_analysed')[0]
			# )
			# plt.savefig('images/' + file.name[:-4] + '_hist.png')
			# plt.close()

			fig = plotly_cdf_curve(df, 'Local density (cells / 100 um^2)', None, title=fname)
			fig.show()

			fig = plotly_rkl_curve(df, 'X', 'Y', title=fname)
			fig.show()
			# exit()
	
	# od = collections.OrderedDict(sorted(data_dict.items()))
	# fig = plotly_boxplot(od, 'Local density (cells / 100 um^2)')
	# fig.show()