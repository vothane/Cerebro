(ns cerebro.DeepBeliefNet
  (:use [cerebro.HiddenLayer.HLayer]
        [cerebro.RestrictedBoltzmannMachine.RBM]
        [cerebro.LogisticRegression.LogReg]
        [cerebro.Utils.utils]))

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
                    activate (fn [inputs weights] (mapv #(dot-product inputs %) weights))
                    output (mapv #(activate linear-output (:weights (logreg->map %))) log-layers)
                    bias-out (mapv #(mapv + %1 (:bias (logreg->map %2))) output log-layers)]
                (mapv #(softmax %) bias-out)))

   :->map {:sigmoid-layers sigmoid-layers :rbm-layers rbm-layers :log-layers log-layers}
  })  
   
    (declare sample-hidden-layers)
    (declare cycle-epochs)
    (declare pretrain-rbms)

    ;; helper functions for DBN
    
    (defn contrast-diverge-rbms [rbms hidden-layers X-train epochs lr k]
      (let [inputs (sample-hidden-layers (drop-last hidden-layers) X-train)] 
        (pretrain-rbms rbms inputs epochs lr k)))
  
    (defn train-logs [logs hidden-layers X-train Y-train epochs lr]
      (let [inputs (sample-hidden-layers hidden-layers X-train)
            train-log (fn [log] ((:train log) inputs Y-train lr))] 
        (mapv #(cycle-epochs % epochs train-log) logs)))
       

        (declare sample-inputs)

        ;; helper functions for DBN helper functions

        (defn pretrain-rbms [rbms inputs epochs lr k]
          (let [con-div (fn [rbm input] (contrastive-divergence rbm input lr k))
                con-divs (fn [inputs rbm] (reduce #(con-div %1 %2) rbm inputs))]
            (mapv #(cycle-epochs %1 epochs (partial con-divs %2)) rbms inputs)))

        (defn sample-hidden-layers [hidden-layers init-input]
          (reduce 
            (fn [inputs sigmoid-layer] 
              (conj inputs (sample-inputs sigmoid-layer (last inputs)))) 
            [init-input] 
            hidden-layers))
        
        (defn sample-inputs [sigmoid-layer inputs]
          (mapv #(sigmoid-sample-h-given-v sigmoid-layer %) inputs))

        (defn cycle-epochs [domain epochs f] 
          (reduce (fn [domain _] (f domain)) domain (range epochs)))
        