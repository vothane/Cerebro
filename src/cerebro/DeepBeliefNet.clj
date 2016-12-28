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


;; API functions
(defn predict [dbn x] ((:predict dbn) x))
(defn pretrain [dbn X-train epochs lr k] ((:pretrain dbn) X-train epochs lr k))
(defn finetune [dbn X-train Y-train epochs lr] ((:finetune dbn) X-train Y-train epochs lr))
(defn dbn->map [dbn] (:->map dbn))


(declare contrast-diverge-rbms)
(declare train-logs)

(defn DBN [sigmoid-layers rbm-layers log-layers] 
  {:pretrain (fn [X-train epochs lr k]
               (let [rbms (contrast-diverge-rbms rbm-layers sigmoid-layers X-train epochs lr k)]
                 (DBN sigmoid-layers rbms log-layers)))

   :finetune (fn finetune [X-train Y-train epochs lr]
               (let [logs (train-logs log-layers sigmoid-layers X-train Y-train epochs lr)]
                 (DBN sigmoid-layers rbm-layers logs)))

   :predict (fn predict [x]
              (let [linear-output (reduce (fn [input layer] (activation layer input))
                                          (activation (first sigmoid-layers) x)
                                          (rest sigmoid-layers))
                    activate (fn [inputs weights] (reduce + (map * inputs weights)))
                    output (map #(activate linear-output (:weights (logreg->map %))) log-layers)
                    _ (println "-----------output-----------") _ (println linear-output) _ (println "-----------output-----------")
                    bias-out (map + output (:bias log-layers))]
                (softmax bias-out)))

   :->map {sigmoid-layers sigmoid-layers :rbm-layers rbm-layers :log-layers log-layers}
  })  
   
    (declare sample-inputs)
    (declare cycle-epochs)

    ;; helper functions for DBN
    
    (defn contrast-diverge-rbms [rbms hidden-layers X-train epochs lr k]
      (let [inputs (sample-inputs hidden-layers X-train)
            con-div (fn [rbm] (contrastive-divergence rbm inputs lr k))] 
        (mapv #(cycle-epochs % epochs con-div) rbms)))
  
    (defn train-logs [logs hidden-layers X-train Y-train epochs lr]
      (let [inputs (sample-inputs hidden-layers X-train)
            train-log (fn [log] ((:train log) inputs Y-train lr))] 
        (mapv #(cycle-epochs % epochs train-log) logs)))
       
        
        (declare sample-hidden-layers)

        ;; helper functions for DBN helper functions
        (defn sample-inputs [hidden-layers train-X]
          (reduce #(conj %1 sample-hidden-layers %2) (vec train-X) hidden-layers))

        (defn cycle-epochs [domain epochs f] 
          (reduce (fn [domain _] (f domain)) domain (range epochs)))

        (defmulti sample-hidden-layers (fn [hidden-layers input] (count hidden-layers)))

        (defmethod sample-hidden-layers 1 [_ input] input)

        (defmethod sample-hidden-layers :default [hidden-layers input]
          (reduce (fn [s-h|v layer] ((:sample-h-given-v layer) s-h|v))
                  ((:sample-h-given-v (first hidden-layers)) input)
                  (rest hidden-layers)))
       