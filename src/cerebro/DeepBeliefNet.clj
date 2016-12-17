(ns cerebro.DeepBeliefNet
  (:use [cerebro.HiddenLayer.HLayer]
        [cerebro.RestrictedBoltzmannMachine.RBM]
        [cerebro.LogisticRegression.LogReg]))

; A deep belief network is obtained by stacking several RBMs on top of each
; other. The hidden layer of the RBM at layer `i` becomes the input of the
; RBM at layer `i+1`. The first layer RBM gets as input the input of the
; network, and the hidden layer of the last RBM represents the output. When
; used for classification, the DBN is treated as a MLP, by adding a logistic
; regression layer on top.

(declare contrast-diverge-rbms)

(defn DBN [W sigs rbms logs] 
  {:shared-weights W
   :sigmoid-layers (reduce #(assoc %1 :weights %2) sigs W)
   :rbm-layers (reduce #(assoc %1 :weights %2) rbms W)
   :log-layers logs

   :pretrain (fn [train-X epochs lr k]
               (let [rbms (contrast-diverge-rbms :rbm-layers :sigmoid-layers train-X epochs lr k)]
                 (DBN (:weights rbms) :sigmoid-layers :rbm-layers :log-layers)))

   :finetune (fn finetune [train-X targets epochs lr]
               (let [data (partition 2 (interleave train-X targets))
                     logl (reduce
                            (fn [log-layer [x y]] (train log-layer (sample-hidden-layers hidden-layers x) y lr))
                            log-layer
                            (take (* (count data) epochs) (cycle data)))]
                 (DBN :shared-weights :sigmoid-layers :rbm-layers logl)))

   :predict (fn predict [dbn x]
              (let [linear-output (reduce (fn [input layer] (activation layer input))
                                          (activation (first :sigmoid-layers) x)
                                          (rest :sigmoid-layers))
                    activate (fn [inputs weights] (reduce + (map * inputs weights)))
                    output (map #(activate linear-output %) (:weights :log-layers))
                    bias-out (map + output (:bias :log-layers))]
                (softmax bias-out))))
  })  
   
    ;; helper functions for DBN
    (defn contrast-diverge-rbms [rbms hidden-layers inputs epochs lr k]
      (map-indexed
        (fn [idx rbm]
          (reduce
            (fn [rbm input] (RBM-contrastive-divergence rbm (sample-hidden-layers (take (inc idx) hidden-layers) input) lr k))
            rbm
            (take (* (count inputs) epochs) (cycle inputs))))
        rbms))
