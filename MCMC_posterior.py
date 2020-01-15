from numpy import *
import numpy as np
from pylab import *
import mypymcLib

#print('Example 1, linear model')
niter=80000

nburn=0e3 #
variables = ['a','b']
labels    = ['$\\alpha$','$\\beta$']

kmin=20000 # cut 1000 chains from the output chain, to have only chains that are converged

xx = np.random.randn(100)*1.0 + 1.0
yy = xx*1.0 + np.random.randn(100)*0.3
yyerr= yy*0.0+0.3 #1.0

def model_prior3(x,pars):      return (pars[0]-1.)**2/0.3**2 + (pars[1]-1.)**2/0.4**2

def model_prior2(x,pars):      return (pars[0]-1.)**2/0.05**2 + (pars[1]-.162)**2/0.1**2

def model_prior_Final(x,pars): return (pars[0]-1.00)**2/0.05**2 + (pars[1]-.050)**2/0.05**2

def model_prior(x,pars):       return (pars[0]-1.056)**2/0.01**2 + (pars[1]+.050)**2/0.05**2

def model_prior_with_correlation(x,pars): 
	a_prior   = 0.98 #1.056 
	b_prior   = 0.05
	rho_ab    = 0.7
	sig_a,sig_b=0.01,0.05
	delta_pars = array([(pars[0] - a_prior),(pars[1] - b_prior )])
	cov_prior = array([                                                                                                                                                     
	[sig_a**2,rho_ab*sig_a*sig_b],
	[rho_ab*sig_a*sig_b,sig_b**2] 
	])
	chi2 = dot(dot(delta_pars,np.linalg.inv(cov_prior)),delta_pars)
	return chi2

def model(x,pars): return pars[0]*x+pars[1]


data = mypymcLib.Data(xvals=xx,yvals=yy,errors=yyerr,model=model)

chains_data    = mypymcLib.run_mcmc([data], variables=variables,niter=niter,nburn=nburn,w_ll_model='test_linear')
nchains_data = mypymcLib.burnChains(chains_data,kmin=kmin)

prior_cmb = mypymcLib.Data(model=model_prior_with_correlation, prior=True)

chains_prior    = mypymcLib.run_mcmc([prior_cmb], variables=variables,niter=niter,nburn=nburn,w_ll_model='test_linear')
nchains_prior = mypymcLib.burnChains(chains_prior,kmin=kmin)

chains_data_prior    = mypymcLib.run_mcmc([data,prior_cmb], variables=variables,niter=niter,nburn=nburn,w_ll_model='test_linear')
nchains_data_prior = mypymcLib.burnChains(chains_data_prior,kmin=kmin)

ion()
figure(1),clf()

smoothparameter = 10.0
mypymcLib.matrixplot(nchains_prior, variables, 'blue', smoothparameter, labels=labels,Blabel='Prior',NsigLim=7,plotScatter=False)
mypymcLib.matrixplot(nchains_data , variables, 'green', smoothparameter, labels=labels,Blabel='Data',NsigLim=7,plotScatter=False)
mypymcLib.matrixplot(nchains_data_prior, variables, 'red', smoothparameter, labels=labels,Blabel='Prior+Data',NsigLim=7,plotScatter=False)

show()


figure(2),clf()
errorbar(xx,yy,yerr=yyerr,fmt='.',label='Data')

m_a_prior       = mean(chains_prior['a'])
m_b_prior       = mean(chains_prior['b'])
m_a_data        = mean(chains_data['a'])
m_b_data        = mean(chains_data['b'])
m_a_data_prior  = mean(chains_data_prior['a'])
m_b_data_prior  = mean(chains_data_prior['b'])

chi2_prior      = sum( ( ( model(xx,[m_a_prior,m_b_prior]) - yy)/yyerr)**2. )
chi2_data       = sum( ( ( model(xx,[m_a_data,m_b_data]) - yy)/yyerr)**2. )
chi2_data_prior = sum( ( ( model(xx,[m_a_data_prior,m_b_data_prior]) - yy)/yyerr)**2. )

plot(xx,model(xx,[m_a_prior,m_b_prior])          ,'b-',label='Prior $\chi^2=$%0.2f / (%0.2f - %0.2f) '%(chi2_prior,len(xx),len(labels)) )
plot(xx,model(xx,[m_a_data,m_b_data])            ,'g-',label='Data $\chi^2=$%0.2f / (%0.2f - %0.2f) '%(chi2_data,len(xx),len(labels)))
plot(xx,model(xx,[m_a_data_prior,m_b_data_prior]),'r-',label='Data + Prior $\chi^2=$%0.2f / (%0.2f - %0.2f) '%(chi2_data_prior,len(xx),len(labels)))

ylabel('y',size=25),xlabel('x',size=25)
legend()
show()



###### calculate all chi2s ######

kmin = len(nchains_prior['a'])
chi2_chain_prior      = zeros(kmin)
chi2_chain_data 	  = zeros(kmin)
chi2_chain_data_prior = zeros(kmin)

for i_chain in range(kmin):
	chi2_chain_prior[i_chain]      = -2.*data([nchains_prior['a'][i_chain],nchains_prior['b'][i_chain]])
	chi2_chain_data[i_chain]       = -2.*data([nchains_data['a'][i_chain],nchains_data['b'][i_chain]])
	chi2_chain_data_prior[i_chain] = -2.*data([nchains_data_prior['a'][i_chain],nchains_data_prior['b'][i_chain]])

figure(4),clf()
subplot(1,2,1), plot(nchains_prior['a'],chi2_chain_prior,'.',label='prior'),xlabel(labels[0]),ylabel('$\chi^2$')
subplot(1,2,2), plot(nchains_prior['b'],chi2_chain_prior,'.',label='prior'),xlabel(labels[1]),ylabel('$\chi^2$')
draw(),legend()

"""
##### Do three dimensional curve:
from mpl_toolkits import mplot3d
fig = plt.figure(7)
ax = plt.axes(projection='3d')
wchi2ok = where(chi2_chain_prior<90)
xin = nchains_prior['a'][wchi2ok]
yin = nchains_prior['b'][wchi2ok]
zin = chi2_chain_prior[wchi2ok]
ax.scatter(xin, yin, zin, c=zin, cmap='viridis', linewidth=0.5);
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('$\chi^2$')
##### End: Do three dimensional curve:
"""

#####

figure(5),clf()
subplot(121), plot(nchains_data['a'],chi2_chain_data,'.',label='data'),xlabel(labels[0]),ylabel('$\chi^2$')
subplot(122), plot(nchains_data['b'],chi2_chain_data,'.',label='data'),xlabel(labels[1]),ylabel('$\chi^2$')
draw(),legend()

figure(6),clf()
subplot(121), plot(nchains_data_prior['a'],chi2_chain_data_prior,'.',label='data+prior'),xlabel(labels[0]),ylabel('$\chi^2$')
subplot(122), plot(nchains_data_prior['b'],chi2_chain_data_prior,'.',label='data+prior'),xlabel(labels[1]),ylabel('$\chi^2$')
draw(),legend()



'''
print('Example 2, quadratic model')
Ex2_niter=80000

Ex2_nburn=0e3 #
Ex2_variables = ['a','b','c','d']
Ex2_labels    = ['$\\alpha$','$\\beta$','$\\gamma$','$\\delta$']

Ex2_kmin=20000 # cut 1000 chains from the output chain, to have only chains that are converged

Ex2_xx = np.random.randn(100)*1.0 + 1.0
Ex2_yy = 1.0*(Ex2_xx*1.0 + np.random.randn(100)*0.3)**3 + 1.0*(Ex2_xx*1.0 + np.random.randn(100)*0.3)**2 + 1.0*(Ex2_xx*1.0 + np.random.randn(100)*0.3)**1 + 1.0*(Ex2_xx*1.0 + np.random.randn(100)*0.3)**0  
Ex2_yyerr= Ex2_yy*0.0+0.3 #1.0

def Ex2_model_prior(x,pars):       return (pars[0]-1.0)**2/0.01**2 + (pars[1]-1.0)**2/0.05**2 + (pars[2]-1.0)**2/0.05**2 + (pars[3]-1.0)**2/0.05**2

def Ex2_model_prior_with_correlation(x,pars): 
	a_prior   = 0.98 #1.056 
	b_prior   = 0.05
	rho_ab    = 0.7
	sig_a,sig_b=0.01,0.05
	delta_pars = array([(pars[0] - a_prior),(pars[1] - b_prior )])
	cov_prior = array([                                                                                                                                                     
	[sig_a**2,rho_ab*sig_a*sig_b],
	[rho_ab*sig_a*sig_b,sig_b**2] 
	])
	chi2 = dot(dot(delta_pars,np.linalg.inv(cov_prior)),delta_pars)
	return chi2

def Ex2_model(x,pars): return pars[0]*x**3.+pars[1]*x**2.+pars[2]*x**1.+pars[3]


Ex2_data = mypymcLib.Data(xvals=Ex2_xx,yvals=Ex2_yy,errors=Ex2_yyerr,model=Ex2_model)

Ex2_chains_data    = mypymcLib.run_mcmc([Ex2_data], variables=Ex2_variables,niter=Ex2_niter,nburn=Ex2_nburn,w_ll_model='test_quadratic')
Ex2_nchains_data = mypymcLib.burnChains(Ex2_chains_data,kmin=Ex2_kmin)

Ex2_prior_cmb = mypymcLib.Data(model=Ex2_model_prior_with_correlation, prior=True)

Ex2_chains_prior    = mypymcLib.run_mcmc([Ex2_prior_cmb], variables=Ex2_variables,niter=Ex2_niter,nburn=Ex2_nburn,w_ll_model='test_quadratic')
Ex2_nchains_prior = mypymcLib.burnChains(Ex2_chains_prior,kmin=Ex2_kmin)

Ex2_chains_data_prior    = mypymcLib.run_mcmc([Ex2_data,Ex2_prior_cmb], variables=Ex2_variables,niter=Ex2_niter,nburn=Ex2_nburn,w_ll_model='test_quadratic')
Ex2_nchains_data_prior = mypymcLib.burnChains(Ex2_chains_data_prior,kmin=Ex2_kmin)


ion(),figure(7),clf()

smoothparameter = 10.0
mypymcLib.matrixplot(Ex2_nchains_prior, Ex2_variables, 'blue', smoothparameter, labels=Ex2_labels,Blabel='Prior',NsigLim=7,plotScatter=False)
mypymcLib.matrixplot(Ex2_nchains_data , Ex2_variables, 'green', smoothparameter, labels=Ex2_labels,Blabel='Data',NsigLim=7,plotScatter=False)
mypymcLib.matrixplot(Ex2_nchains_data_prior, Ex2_variables, 'red', smoothparameter, labels=Ex2_labels,Blabel='Prior+Data',NsigLim=7,plotScatter=False)

draw(),show()


figure(8),clf()
errorbar(Ex2_xx,Ex2_yy,yerr=Ex2_yyerr,fmt='.',label='Data')

Ex2_m_a_prior       = mean(Ex2_chains_prior[0]['a'])
Ex2_m_b_prior       = mean(Ex2_chains_prior[0]['b'])
Ex2_m_c_prior       = mean(Ex2_chains_prior[0]['c'])
Ex2_m_d_prior       = mean(Ex2_chains_prior[0]['d'])
Ex2_m_a_data        = mean(Ex2_chains_data[0]['a'])
Ex2_m_b_data        = mean(Ex2_chains_data[0]['b'])
Ex2_m_c_data        = mean(Ex2_chains_data[0]['c'])
Ex2_m_d_data        = mean(Ex2_chains_data[0]['d'])
Ex2_m_a_data_prior  = mean(Ex2_chains_data_prior[0]['a'])
Ex2_m_b_data_prior  = mean(Ex2_chains_data_prior[0]['b'])
Ex2_m_c_data_prior  = mean(Ex2_chains_data_prior[0]['c'])
Ex2_m_d_data_prior  = mean(Ex2_chains_data_prior[0]['d'])

Ex2_chi2_prior      = sum( ( ( Ex2_model(Ex2_xx,[Ex2_m_a_prior,Ex2_m_b_prior,Ex2_m_c_prior,Ex2_m_d_prior]) - Ex2_yy)/Ex2_yyerr)**2. )
Ex2_chi2_data       = sum( ( ( Ex2_model(Ex2_xx,[Ex2_m_a_data,Ex2_m_b_data,Ex2_m_c_data,Ex2_m_d_data]) - Ex2_yy)/Ex2_yyerr)**2. )
Ex2_chi2_data_prior = sum( ( ( Ex2_model(Ex2_xx,[Ex2_m_a_data_prior,Ex2_m_b_data_prior,Ex2_m_c_data_prior,Ex2_m_d_data_prior]) - Ex2_yy)/Ex2_yyerr)**2. )

plot(Ex2_xx,Ex2_model(Ex2_xx,[Ex2_m_a_prior,Ex2_m_b_prior,Ex2_m_c_prior,Ex2_m_d_prior])          ,'b.',label='Prior $\chi^2=$%0.2f / (%0.2f - %0.2f) '%(Ex2_chi2_prior,len(Ex2_xx),len(Ex2_labels)) )
plot(Ex2_xx,Ex2_model(Ex2_xx,[Ex2_m_a_data,Ex2_m_b_data,Ex2_m_c_data,Ex2_m_d_data])            ,'g.',label='Data $\chi^2=$%0.2f / (%0.2f - %0.2f) '%(Ex2_chi2_data,len(Ex2_xx),len(Ex2_labels)))
plot(Ex2_xx,Ex2_model(Ex2_xx,[Ex2_m_a_data_prior,Ex2_m_b_data_prior,Ex2_m_c_data_prior,Ex2_m_d_data_prior]),'r.',label='Data + Prior $\chi^2=$%0.2f / (%0.2f - %0.2f) '%(Ex2_chi2_data_prior,len(Ex2_xx),len(Ex2_labels)))

ylabel('y',size=25),xlabel('x',size=25)
legend()
show()



###### calculate all chi2s ######

Ex2_kmin = len(Ex2_nchains_prior[0]['a'])
Ex2_chi2_chain_prior      = zeros(Ex2_kmin)
Ex2_chi2_chain_data 	  = zeros(Ex2_kmin)
Ex2_chi2_chain_data_prior = zeros(Ex2_kmin)

for i_chain in range(Ex2_kmin):
	Ex2_chi2_chain_prior[i_chain]      = -2.*Ex2_data([Ex2_nchains_prior['a'][i_chain],Ex2_nchains_prior['b'][i_chain],Ex2_nchains_prior['c'][i_chain],Ex2_nchains_prior['d'][i_chain]])
	Ex2_chi2_chain_data[i_chain]       = -2.*Ex2_data([Ex2_nchains_data['a'][i_chain],Ex2_nchains_data['b'][i_chain],Ex2_nchains_data['c'][i_chain],Ex2_nchains_data['d'][i_chain]])
	Ex2_chi2_chain_data_prior[i_chain] = -2.*Ex2_data([Ex2_nchains_data_prior['a'][i_chain],Ex2_nchains_data_prior['b'][i_chain],Ex2_nchains_data_prior['c'][i_chain],Ex2_nchains_data_prior['d'][i_chain]])

figure(9),clf()
subplot(1,2,1), plot(Ex2_nchains_prior['a'],Ex2_chi2_chain_prior,'.',label='prior'),xlabel(Ex2_labels[0]),ylabel('$\chi^2$')
subplot(1,2,2), plot(Ex2_nchains_prior['b'],Ex2_chi2_chain_prior,'.',label='prior'),xlabel(Ex2_labels[1]),ylabel('$\chi^2$')
draw(),legend()

"""
##### Do three dimensional curve:
from mpl_toolkits import mplot3d
fig = plt.figure(7)
ax = plt.axes(projection='3d')
wchi2ok = where(chi2_chain_prior<90)
xin = nchains_prior['a'][wchi2ok]
yin = nchains_prior['b'][wchi2ok]
zin = chi2_chain_prior[wchi2ok]
ax.scatter(xin, yin, zin, c=zin, cmap='viridis', linewidth=0.5);
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('$\chi^2$')
##### End: Do three dimensional curve:
"""

#####

figure(10),clf()
subplot(121), plot(Ex2_nchains_data['a'],Ex2_chi2_chain_data,'.',label='data'),xlabel(Ex2_labels[0]),ylabel('$\chi^2$')
subplot(122), plot(Ex2_nchains_data['b'],Ex2_chi2_chain_data,'.',label='data'),xlabel(Ex2_labels[1]),ylabel('$\chi^2$')
draw(),legend()

figure(11),clf()
subplot(121), plot(Ex2_nchains_data_prior['a'],Ex2_chi2_chain_data_prior,'.',label='data+prior'),xlabel(Ex2_labels[0]),ylabel('$\chi^2$')
subplot(122), plot(Ex2_nchains_data_prior['b'],Ex2_chi2_chain_data_prior,'.',label='data+prior'),xlabel(Ex2_labels[1]),ylabel('$\chi^2$')
draw(),legend()
'''
