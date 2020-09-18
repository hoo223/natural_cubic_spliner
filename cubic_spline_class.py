import numpy as np

class cubic_spliner:
    def __init__(self):
        pass
    
    def jacobi(self, A, b, x0, tol, n_iterations=300):
        """
        Performs Jacobi iterations to solve the line system of
        equations, Ax=b, starting from an initial guess, ``x0``.

        Returns:
        x, the estimated solution
        """

        n = A.shape[0]
        x = x0.copy()
        x_prev = x0.copy()
        counter = 0
        x_diff = tol+1

        while (x_diff > tol) and (counter < n_iterations): #iteration level
            for i in range(0, n): #element wise level for x
                s = 0
                for j in range(0,n): #summation for i !=j
                    if i != j:
                        s += A[i,j] * x_prev[j] 

                x[i] = (b[i] - s) / A[i,i]
            #update values
            counter += 1
            x_diff = (np.sum((x-x_prev)**2))**0.5 
            x_prev = x.copy() #use new x for next iteration


        print("Number of Iterations: ", counter)
        print("Norm of Difference: ", x_diff)
        return x

    def cubic_spline(self, x, y, tol = 1e-100):
        """
        Interpolate using natural cubic splines.

        Generates a strictly diagonal dominant matrix then applies Jacobi's method.

        Returns coefficients:
        b, coefficient of x of degree 1
        c, coefficient of x of degree 2
        d, coefficient of x of degree 3
        """ 
        x = np.array(x)
        y = np.array(y)
        ### check if sorted
        if np.any(np.diff(x) < 0):
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

        size = len(x)
        delta_x = np.diff(x)
        delta_y = np.diff(y)

        ### Get matrix A
        A = np.zeros(shape = (size,size))
        b = np.zeros(shape=(size,1))
        A[0,0] = 1
        A[-1,-1] = 1

        for i in range(1,size-1):
            A[i, i-1] = delta_x[i-1]
            A[i, i+1] = delta_x[i]
            A[i,i] = 2*(delta_x[i-1]+delta_x[i])
        ### Get matrix b
            b[i,0] = 3*(delta_y[i]/delta_x[i] - delta_y[i-1]/delta_x[i-1])

        ### Solves for c in Ac = b
        print('Jacobi Method Output:')
        c = self.jacobi(A, b, np.zeros(len(A)), tol = tol, n_iterations=1000)

        ### Solves for d and b
        d = np.zeros(shape = (size-1,1))
        b = np.zeros(shape = (size-1,1))
        for i in range(0,len(d)):
            d[i] = (c[i+1] - c[i]) / (3*delta_x[i])
            b[i] = (delta_y[i]/delta_x[i]) - (delta_x[i]/3)*(2*c[i] + c[i+1])    

        return b.squeeze(), c.squeeze(), d.squeeze()
    
    # 구해진 polynomial parameter를 이용해 polynomial 식 구현 (1,2차 미분 포함)
    def cubic_polynomial(self, xk, yk, bk, ck, dk, x):
        delta_x = x-xk
        x = yk + bk*delta_x + ck*delta_x*delta_x + dk*delta_x*delta_x*delta_x
        dx = bk + 2*ck*delta_x + 3*dk*delta_x*delta_x
        ddx = 2*ck + 6*dk*delta_x
        return x, dx, ddx
    
    def check_poly_num(self, t, ts): # 어느 구간의 polynomial에 해당하는지 
        num=0
        for i in range(len(ts)):
            if t>=ts[i]:
                num+=1
        return num
    
if __name__=="__main__":
    # Example
    cubic_spliner_object = cubic_spliner()
    
    # path point
    ts = [1, 2, 3, 4, 5]
    xs = [1, 2, 2.8, 3.4, 3.8]
    total_time = ts[len(ts)-1] -ts[0]
    
    time = []
    q = []
    dq = []
    ddq = []
    
    # compute polynomial parameters
    b, c, d = cubic_spliner_object.cubic_spline(ts, xs, 0)

    # compute states
    for i in range(1000):
        t = total_time/1000.0*i+ts[0]
        num = cubic_spliner_object.check_poly_num(t,ts)
        if num < len(ts):
            x, dx, ddx = cubic_spliner_object.cubic_polynomial(ts[num-1], xs[num-1], b[num-1], c[num-1], d[num-1], t)
            time.append(t)
            q.append(x)
            dq.append(dx)
            ddq.append(ddx)
