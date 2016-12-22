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

   :pretrain (fn [X-train epochs lr k]
               (let [rbms (contrast-diverge-rbms :rbm-layers :sigmoid-layers X-train epochs lr k)]
                 (DBN (:weights rbms) :sigmoid-layers rbms :log-layers)))

   :finetune (fn finetune [train-X targets epochs lr]
               (let [logs (train-logs :log-layers :sigmoid-layers X-train Y-train epochs lr)]
                 (DBN :shared-weights :sigmoid-layers :rbm-layers logs)))

   :predict (fn predict [dbn x]
              (let [linear-output (reduce (fn [input layer] (activation layer input))
                                          (activation (first :sigmoid-layers) x)
                                          (rest :sigmoid-layers))
                    activate (fn [inputs weights] (reduce + (map * inputs weights)))
                    output (map #(activate linear-output %) (:weights :log-layers))
                    bias-out (map + output (:bias :log-layers))]
                (softmax bias-out))))
  })  
   
    (declare sample-inputs)
    (declare cycle-epochs)

    ;; helper functions for DBN
    (defn contrast-diverge-rbms [rbms hidden-layers X-train epochs lr k]
      (let [inputs (sample-inputs hidden-layers X-train)
            con-div (fn [rbm] ((:contrastive-divergence rbm) inputs lr k))] 
        (mapv #(cycle-epochs % epochs con-div) rbms)))
  
    (defn train-logs [logs hidden-layers X-train Y-train epochs lr]
      (let [inputs (sample-inputs hidden-layers X-train)
            train-log (fn [log] ((:train log) inputs Y-train lr))] 
        (mapv #(cycle-epochs % epochs train-log) logs)))

        ;; helper functions for DBN helper functions
        (defn sample-inputs [hidden-layers train-X]
          (reduce #(conj %1 sample-hidden-layers %2) (vec train-X) hidden-layers))

        (defn cycle-epochs (fn [domain epochs f] 
          (reduce (fn [domain _] (f domain)) domain (range epochs))))


       