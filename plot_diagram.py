import numpy as np
import colorspace
import matplotlib.pyplot as plt

from fast_jm import g, limit, chiu_limit, unconstrained_limit

plt.rcParams.update({
    "text.usetex": True
})

# def OLD_jm_diagram(colours):
    # """
    # 'colours' should be a list of four colours.
    # """
    # rho = 1000000
    # fig, ax = plt.subplots()
    # c_samples = np.genfromtxt(f'data/rho{rho}-coverage-times.csv')
    # u_samples = np.genfromtxt(f'data/unconstrained-rho{rho}-coverage-times.csv')

    # my_range = np.arange(min(-20,min(c_samples)-0.1,min(u_samples)-0.1),max(50,max(c_samples)+0.1,max(u_samples)+0.1),0.1)

    # c_curve = g(rho,c_samples) # constrained JM
    # u_curve = g(rho,u_samples) # unconstrained
    # c_curve.sort()
    # u_curve.sort()
    # ax.plot(c_curve, (np.arange(c_curve.size)+1)/c_curve.size, colours[0], linewidth=2, label="Empirical distribution of $g(T_\\rho,\\rho)$")
    # c_limit = limit(my_range)
    # ax.plot(my_range, c_limit, colours[1],linestyle='dashed',linewidth=1.5,label="Limiting cdf of $g(T_\\rho,\\rho)$ (from Thm 2.8)")
    # ax.plot(u_curve, (np.arange(u_curve.size)+1)/u_curve.size, colours[-1], linewidth=2, label="Empirical distribution of $g(\\tilde T_\\rho,\\rho)$")
    # u_limit = chiu_limit(my_range, rho)
    # ax.plot(my_range, u_limit, colours[-2],linestyle='dashed',linewidth=1.5,label="Estimated cdf of $g(\\tilde T_\\rho,\\rho)$ (from Chiu 1995)")

    # ax.set_ylim(0,1)
    # ax.set_xlim(-10,40)
    # ax.legend(loc='lower right')
    # ax.set_title(f'Coverage time for JM processes in $[0,1]^2$ with arrival rate $\\rho=10^{int(np.log10(rho))}$')
    # ax.set_xlabel('$\\beta$')
    # ax.set_ylabel('')
    # fig.tight_layout()
    # fig.savefig('jm_diagram.pdf')
    # fig.savefig('jm_diagram.png')
    
def jm_diagram(colours):
    """
    'colours' should be a list of four colours.
    """
    rho = 1000000
    fig, ax = plt.subplots()
    c_samples = np.genfromtxt(f'data/rho{rho}-coverage-times.csv')
    u_samples = np.genfromtxt(f'data/unconstrained-rho{rho}-coverage-times.csv')

    my_range = np.arange(min(-20,min(c_samples)-0.1,min(u_samples)-0.1),max(50,max(c_samples)+0.1,max(u_samples)+0.1),0.1)

    c_curve = g(rho,c_samples) # constrained JM
    u_curve = g(rho,u_samples) # unconstrained
    c_curve.sort()
    u_curve.sort()
    ax.plot(c_curve, (np.arange(c_curve.size)+1)/c_curve.size, colours[0], linewidth=2, label="Empirical distribution of $g(T_\\rho,\\rho)$")
    c_limit = limit(my_range)
    ax.plot(my_range, c_limit, colours[1],linestyle='dashed',linewidth=1.5,label="Limiting cdf of $g(T_\\rho,\\rho)$ (from Thm 2.9)")
    ax.plot(u_curve, (np.arange(u_curve.size)+1)/u_curve.size, colours[-1], linewidth=2, label="Empirical distribution of $g(\\tilde T_\\rho,\\rho)$")
    u_limit = unconstrained_limit(my_range)
    ax.plot(my_range, u_limit, colours[-2],linestyle='dashed',linewidth=1.5,label="Limiting cdf of $g(\\tilde T_\\rho,\\rho)$ (from Prop 2.8)")

    ax.set_ylim(0,1)
    ax.set_xlim(-10,20)
    ax.legend(loc='lower right')
    ax.set_title(f'Coverage time for JM processes in $[0,1]^2$ with arrival rate $\\rho=10^{int(np.log10(rho))}$')
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.savefig('jm_diagram.pdf')
    fig.savefig('jm_diagram.eps')
    fig.savefig('jm_diagram.png')
    plt.close()
    
# def compare_limits(rho,filesuffix):
    # fig, ax = plt.subplots()
    # my_range = np.arange(-10,20,0.1)
    # u_limit = unconstrained_limit(my_range)
    # u_estim = chiu_limit(my_range, rho)
    # ax.plot(my_range, u_limit, "blue",linestyle='dashed',linewidth=1.5,label="Limit from Prop 2.7")
    # ax.plot(my_range, u_estim, "red" ,linestyle='dashed',linewidth=1.5,label="Estimate from Chiu 1995")
    # ax.set_ylim(0,1)
    # ax.set_xlim(-10,20)
    # ax.legend(loc='lower right')
    # ax.set_xlabel('$\\beta$')
    # ax.set_ylabel('')
    # fig.tight_layout()
    # fig.savefig(f'compare_limits_{filesuffix}.png')
    # return
    
if __name__=='__main__':
    plt.rcParams['figure.figsize'] = [10,3.5]
    lblue = -0.4
    lred = 0.4
    r = '#ff0000'
    b = '#0000ff'
    b = colorspace.utils.lighten(b,lblue)
    r = colorspace.utils.lighten(r,lred)
    colours = [r, r, b, b]
    jm_diagram(colours)
    
    # Check that our limit is close to Chiu's
    # for exponent in range(0,19):
        # rho = 10**exponent
        # compare_limits(rho,exponent)
