(ns cerebro.DeepBeliefNet
  (:use [cerebro.HiddenLayer.HLayer]
        [cerebro.RestrictedBoltzmannMachine.RBM]))

; A deep belief network is obtained by stacking several RBMs on top of each
; other. The hidden layer of the RBM at layer `i` becomes the input of the
; RBM at layer `i+1`. The first layer RBM gets as input the input of the
; network, and the hidden layer of the last RBM represents the output. When
; used for classification, the DBN is treated as a MLP, by adding a logistic
; regression layer on top.

(defmulti sample-hidden-layers (fn [hidden-layers input] (count hidden-layers)))

(defmethod sample-hidden-layers 1 [_ input] input)

(defmethod sample-hidden-layers :default [hidden-layers input]
  (reduce (fn [s-h|v layer] (hidden-sample-h|v layer s-h|v))
          (hidden-sample-h|v (first hidden-layers) input)
          (rest hidden-layers)))

(defn contrast-diverge-rbms [rbms hidden-layers inputs epochs lr k]
  (map-indexed
    (fn [idx rbm]
      (reduce
        (fn [rbm input] (RBM-contrastive-divergence rbm (sample-hidden-layers (take (inc idx) hidden-layers) input) lr k))
        rbm
        (take (* (count inputs) epochs) (cycle inputs))))
    rbms))

(defn DBN-pretrain [dbn train-x epochs lr k]
  (let [{hidden-layers :sigmoid-layers rbms :rbm-layers} dbn
        rbms (contrast-diverge-rbms rbms hidden-layers train-x epochs lr k)]
    (assoc dbn :rbm-layers (vec rbms))))
