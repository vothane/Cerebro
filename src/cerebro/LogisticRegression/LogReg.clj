; Logistic regression is a probabilistic, linear classifier. It is parametrized
; by a weight matrix :math:`weights` and a bias vector :math:`bias`. 
; Classification is done by projecting data points onto a set of hyperplanes, 
; the distance to which is used to determine a class membership probability.

(defrecord LogReg [N
	               num-inputs
	               num-outputs
	               weights
	               bias])

(defn make-log-reg [n n-in n-out]
  (->LogReg n 
            n-in 
            n-out 
            (partition n-out (take (* n-out n-in) (repeat 0.0))) 
            (take n-out (repeat 0.0))))	           
