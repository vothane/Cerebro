(ns cerebro.DeepBeliefNet)

; A deep belief network is obtained by stacking several RBMs on top of each
; other. The hidden layer of the RBM at layer `i` becomes the input of the
; RBM at layer `i+1`. The first layer RBM gets as input the input of the
; network, and the hidden layer of the last RBM represents the output. When
; used for classification, the DBN is treated as a MLP, by adding a logistic
; regression layer on top.

(defrecord DBN [N
	              num-inputs
	              hidden-layer-sizes
	              num-outputs
	              num-layers
	              sigmoid-layers
	              rbm-layers
	              log-layer])

(defn make-dbn [n n-ins hlayer-sizes n-outs n-layers]
  (let [sigmoid-layers (map #(make-hidden-layer n %1 %2) 
                         (assoc (vec hlayer-sizes) 0 0)
                         hlayer-sizes)]
    (->DBN n 
           n-ins 
           hlayer-sizes
           n-outs
           n-layers
           sigmoid_layers
           (map #(make-rbm n %1 %2 (:weights %3) (:bias %3)) 
             (assoc (vec hlayer-sizes) 0 0)
             hlayer-sizes
             sigmoid-layers)
           (make-log-reg n (last hlayer-sizes) n-outs))))
     