# KI2022 Tutorial: Universal Differential Equations (UDE) in Julia

|          |                                    |
| -------- | ---------------------------------- |
| Speaker  | Stephan Sahm stephan.sahm@jolin.io |
| Company  | [Jolin.io](https://www.jolin.io)   |
| Short    | UDE                                |
| Day      | Monday 10th September              |
| Time     | 09:00 - 12:00 CEST                 |
| Duration | 3h                                 |

This is the material for the UDE tutorial of the conference [KI 2022](https://ki2022.gi.de/).

## Abstract

Let's merge domain-specific models with generic learning. With Julia it is easy to combine these different worlds, differential equations on the one hand and (deep) neural networks on the other. The combination is known as Universal Differential Equations, and published in the paper “Universal Differential Equations for Scientific Machine Learning” by Rackauckas et al. in November 2021. This new modelling flexibility of scientific machine learning brings immense benefits to all kinds of computational sciences.

We will explain the theory behind Universal Differential Equations, including the subclass of Neural Differential Equations, why Julia is particularly suited to this, and see the power of such models in practice.

Everything is interactive and beginner-friendly, with the opportunity to recreate such state-of-the-art models live.

## Prerequisites
- it is good to have at least minimal programming experience in one language
- no knowledge of Julia is required
- knowledge of differential equations is benefitial, but not necessary, as everything will be hands-on with real code

## Content

The tutorial consists of a collection of jupyter notebooks. You can run them and get you individual interactive development environment by just clicking on the respective part
1. [introduction to julia](https://mybinder.org/v2/gh/jolin-io/KI2022-tutorial-universal-differential-equations/main?filepath=01%20introduction%20to%20julia.ipynb)
2. [introduction to deep learning in julia](https://mybinder.org/v2/gh/jolin-io/KI2022-tutorial-universal-differential-equations/main?filepath=02%20introduction%20to%20deep%20learning%20in%20julia.ipynb)
3. [deep dive into universal differential equations](https://mybinder.org/v2/gh/jolin-io/KI2022-tutorial-universal-differential-equations/main?filepath=03%20deep%20dive%20into%20universal%20differential%20equations.ipynb)
4. [introduction to bayesian differential equations](https://mybinder.org/v2/gh/jolin-io/KI2022-tutorial-universal-differential-equations/main?filepath=04%20introduction%20to%20bayesian%20differential%20equations.ipynb)

## Local installation (usually not needed)

If the links above don't work for you, you can run the tutorial locally instead of relying on mybinder.org.

This process is identical to how mybinder.org is actually doing it. Hence you will get the very same environment.

1. Install [docker](https://docs.docker.com/get-docker/)

2. Install [repo2docker](https://repo2docker.readthedocs.io/en/latest/install.html) by running

    ```bash
    python3 -m pip install --user jupyter-repo2docker
    ```

    If you do not have python, consider installing it via [Anaconda](https://www.anaconda.com/products/individual).

3. Execute repo2docker on this repository. It will take several 10 minutes to build everything.

    ```bash
    jupyter-repo2docker https://github.com/jolin-io/KI2022-tutorial-universal-differential-equations
    ```

    Usually, a browser is opened automatically for you, but if not, an url is also printed at the very end of the command output. Copy that one to your browser and you are ready to go.

<br>

## Supported by [Jolin.io](https://www.jolin.io)
Data science consultancy with focus on the Julia language and high performance scientific computing.
![](https://www.jolin.io/assets/Jolin/Jolin-Banner-Website-v1.1-darkmode.webp)

