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

(defn sync-layers [to from]
  (mapv #(assoc %1 :weights (:weights %2) :bias (:hbias %2)) to from))

(defn DBN-pretrain [dbn train-data epochs lr k]
  (let [{hidden-layers :sigmoid-layers rbms :rbm-layers} dbn
        rbms (contrast-diverge-rbms rbms hidden-layers train-data epochs lr k)]
    (assoc dbn :rbm-layers (vec rbms) :sigmoid-layers (sync-layers hidden-layers rbms))))

(defn finetune [dbn train-data targets epochs lr]
  (let [{hidden-layers :sigmoid-layers log-layer :log-layer} dbn
        data (partition 2 (interleave train-data targets))
        logl (reduce
               (fn [log-layer [x y]] (train log-layer (sample-hidden-layers hidden-layers x) y lr))
               log-layer
               (take (* (count data) epochs) (cycle data)))]
    (assoc dbn :log-layer logl)))

(defn predict [dbn x]
  (let [{sigmoid-layers :sigmoid-layers log-layer :log-layer} dbn
        linear-output (reduce (fn [input layer] (activation layer input))
                              (activation (first sigmoid-layers) x)
                              (rest sigmoid-layers))
        activate      (fn [inputs weights] (reduce + (map * inputs weights)))
        output        (map #(activate linear-output %) (:weights log-layer))
        bias-out      (map + output (:bias log-layer))]
    (softmax bias-out)))

