(ns cerebro.DBN_test
    (:require [clojure.test :refer :all]
              [cerebro.DeepBeliefNet :refer :all])
    (:use [cerebro.HiddenLayer.HLayer]
          [cerebro.RestrictedBoltzmannMachine.RBM]
          [cerebro.LogisticRegression.LogReg]))

;; weights to be shared by hidden layers and rbm layers through "structural sharing"
(def weights (list 
               [[ 0.14839871643137212 -0.08501163823540675 0.051985421731801734 -0.14855205346676653 -0.044137597789180485 -0.0701731889478024] 
                [-0.10252046344168927  0.05177738360494413 0.132389904383267    -0.11088185248031387 -0.07047144939818163   0.1342016154235016] 
                [ 0.11659342725429114 -0.07565106509550569 0.036360063996785374 -0.08211466451905437  0.09155141306199341  -0.16083974594778425]] 
               [[ 0.1913826375359961  0.19959579863963633  -0.09572763885651328] 
                [-0.04920529688188052 0.006828221921254152 -0.17304539780927708] 
                [-0.1938654179145095  0.1287133626717507   -0.06540140494644048]]))

(def sigmoid-layers [(HiddenLayer (first weights) [0 0 0]) 
                     (HiddenLayer (last weights) [0 0 0])])

(def rbm-layers [(RBM (first weights) [0 0 0] [0 0 0 0 0 0] 6) 
                 (RBM (last weights) [0 0 0] [0 0 0] 6)])

(def log-layer [(LogReg [[0 0 0] [0 0 0]] [0 0] 6)])

(def dbn (DBN sigmoid-layers rbm-layers log-layer))

(def training-features
  [[1 1 1 0 0 0]
   [1 0 1 0 0 0]
   [1 1 1 0 0 0]
   [0 0 1 1 1 0]
   [0 0 1 1 0 0]
   [0 0 1 1 1 0]])

(def training-predictors
  [[1 0]
   [1 0]
   [1 0]
   [0 1]
   [0 1]
   [0 1]])

(def unknown-features-X
  [[1 1 0 0 0 0]
   [1 1 1 1 0 0]
   [0 0 0 1 1 0]
   [0 0 1 1 1 0]])
                    
(def outcomes-Y
  [[1 0]
   [1 0]
   [1 0]
   [0 1]
   [0 1]
   [0 1]])

(def W1 (list 
         [[3.830040562610957 2.2572438689428482 1.1291324614561007 -4.0898381940839315 -2.9943059764521114 -1.4455110000150806] 
          [-0.23531638963814172 -0.7089522379361282 2.0373878851804412 0.5388205984585597 0.00909457901571344 -1.4625908870311244] 
          [-2.9787234309856543 -2.5183288604959335 0.9839791764425729 2.7849715512720685 1.579416020412225 -1.4473204127308446]] 
         [[2.2679914975465905 -0.032221378861459635 -2.3239115409314617] 
          [0.7196361212711985 0.3531154380197722 -0.698710183627769] 
          [-3.856656951090344 1.1713952667178291 3.7199477886071834]])) 
               
(def sigs1 [(HiddenLayer (first W1) [-0.5889345615736271 0.21725896752865348 -0.5106834878093905])
           (HiddenLayer (last W1) [0.026740498859828266 -0.09725689788213573 -0.9354588375656511])])

(def log1 [(LogReg [[1.362310331208894 0.4118738320958368 -2.1298562224417825] 
                    [-1.362310331208894 -0.4118738320958366 2.1298562224417825]] 
                   [0.17633688988848428 -0.17633688988848387]
                  6)])

(deftest DBN-predict-test
  (testing "DBN should make accurate predictions with mocked sigmoid and log layers"
    (let [dbn (DBN sigs1 rbm-layers log1)]
      (is (= (predict dbn [1 1 0 0 0 0]) [[0.9653090289791912 0.03469097102080875]])))))

(deftest DBN-test
  (testing "DBN should make accurate predictions after training"
    (let [dbn (DBN sigs1 rbm-layers log1)
          dbn (pretrain dbn training-features 1000 0.1 1)
          _ (println "\n-------------------\n" (dbn->map dbn) "\n-------------------\n")
          dbn (finetune dbn training-features training-predictors 500 0.1)]
      (is (= (predict dbn [1 1 0 0 0 0]) [[0.9653090289791912 0.03469097102080875]])))))

(def W2 (list  
          [[3.830040562610957 2.2572438689428482 1.1291324614561007 -4.0898381940839315 -2.9943059764521114 -1.4455110000150806] 
           [-0.23531638963814172 -0.7089522379361282 2.0373878851804412 0.5388205984585597 0.00909457901571344 -1.4625908870311244] 
           [-2.9787234309856543 -2.5183288604959335 0.9839791764425729 2.7849715512720685 1.579416020412225 -1.4473204127308446]] 
          [[2.2679914975465905 -0.032221378861459635 -2.3239115409314617] 
           [0.7196361212711985 0.3531154380197722 -0.698710183627769] 
           [-3.856656951090344 1.1713952667178291 3.7199477886071834]]))

(def init-log-layer (LogReg [[0 0 0] [0 0 0]] [0 0] 6))

(def log-layer (LogReg [[0.06358695635437592 0.03800054523717951 -0.06642946580979513] 
                        [-0.06358695635437592 -0.0380005452371795 0.06642946580979511]] 
                       [-0.0028405655699967128 0.0028405655699967145] 
                       6))
               
(def sigs2 [(HiddenLayer (first W2) [-0.5889345615736271 0.21725896752865348 -0.5106834878093905])
           (HiddenLayer (last W2) [0.026740498859828266 -0.09725689788213573 -0.9354588375656511])])
           
(deftest DBN-finetune-test
  (testing "DBN should finetune correctly"
    (let [dbn (DBN sigs2 rbm-layers init-log-layer)
          X (transient [[0 0 1] [0 1 1] [0 0 1] [1 1 0] [1 1 0] [0 1 0] [0 0 1] [0 0 1] [0 0 1] [1 1 0] [1 1 1] [1 1 0] [0 0 1] [0 1 1] [0 1 1] [1 1 0] [1 0 0] [1 1 0]])]
      (with-redefs-fn {#'sample-inputs (fn [a b]
                                         (let [x (nth X (dec (count X)))
                                               _ (when-not (= 0 (count X)) (pop! X))]
                                           x))}   
        #(is (= (:log-layers (dbn->map (finetune dbn training-features training-predictors 3 0.1))) log-layer))))))
       
(def rbm_layers [{:weights [[0.14562019391751418 -0.0968668410082278 0.12928982106962789 -0.12547145294282122 -0.06287634580550719 -0.1276762491546935] 
                            [-0.1073716648783144 0.035961347345176434 0.20701533477416417 -0.0835437896589793 -0.08824406167239454 0.07400542020688286] 
                            [0.11128084738472974 -0.08912066197313183 0.11570790594024581 -0.05563178054477573 0.07494515594238656 -0.21727459705672983]] 
                  :hbias [-0.0008170355686793586 -0.05106960792287979 0.01481841442320608] 
                  :vbias [-0.016666666666666666 -0.03333333333333333 0.15 0.05 -0.03333333333333334 -0.11666666666666665]
                  :n 6} 
                 {:weights [[0.2188254191505762 0.19886731102242736 -0.12183670839881038] 
                            [-0.023957901208094014 0.007672140879020508 -0.19605154622136878] 
                            [-0.1714998260588481 0.12966750604430738 -0.08935435698494029]] 
                  :hbias [0.07221119555943976 0.05591005066079086 0.035016539502311685] 
                  :vbias [0.05 -6.938893903907228e-18 -0.05]
                  :n 6}])

(def sigm_layers [{:weights [[0.14562019391751418 -0.0968668410082278 0.12928982106962789 -0.12547145294282122 -0.06287634580550719 -0.1276762491546935] 
                           [-0.1073716648783144 0.035961347345176434 0.20701533477416417 -0.0835437896589793 -0.08824406167239454 0.07400542020688286] 
                           [0.11128084738472974 -0.08912066197313183 0.11570790594024581 -0.05563178054477573 0.07494515594238656 -0.21727459705672983]] 
                   :bias [-0.0008170355686793586 -0.05106960792287979 0.01481841442320608]} 
                  {:weights [[0.2188254191505762 0.19886731102242736 -0.12183670839881038] 
                             [-0.023957901208094014 0.007672140879020508 -0.19605154622136878] 
                             [-0.1714998260588481 0.12966750604430738 -0.08935435698494029]] 
                   :bias [0.07221119555943976 0.05591005066079086 0.035016539502311685]}])
               
(deftest DBN-pretrain-test
  (testing "DBN should pretrain correctly"
    (let [dbn (DBN sigs1 rbm-layers log1)
          X (transient [[0 1 0] [0 1 0] [1 0 1] [1 1 1] [1 0 0] [1 1 0] [0 0 1] [1 0 0] [0 1 0] [1 0 0] [0 0 1] [0 1 1] [0 0 0] [0 1 1] [1 0 0] [1 1 0] [0 0 1] [0 1 0] [0 0 1 1 1 0] [0 0 1 1 0 0] [0 0 1 1 1 0] [1 1 1 0 0 0] [1 0 1 0 0 0] [1 1 1 0 0 0] [0 0 1 1 1 0] [0 0 1 1 0 0] [0 0 1 1 1 0] [1 1 1 0 0 0] [1 0 1 0 0 0] [1 1 1 0 0 0] [0 0 1 1 1 0] [0 0 1 1 0 0] [0 0 1 1 1 0] [1 1 1 0 0 0] [1 0 1 0 0 0] [1 1 1 0 0 0]])]
      (with-redefs-fn {#'sample-inputs (fn [a b]
                                         (let [x (nth X (dec (count X)))
                                               _ (when-not (= 0 (count X)) (pop! X))]
                                           x))}   
        #(is (= (:rbm-layers (dbn->map (pretrain dbn training-features 3 0.1 1))) 
                rbm_layers))))))

(deftest hidden-layer-test
  (testing "hidden layer"
    (let [hl (HiddenLayer 
               [[0.14142958607095213 -0.09266647551292541 0.0772905918794082 -0.14099685447053134 -0.05349333511775113 -0.09567588113501038] 
                [-0.11125978685596784 0.04268112877616282 0.15627993581732436 -0.10238492086070262 -0.0792493993065675 0.10793664553091545] 
                [0.10951416049204235 -0.08316652554536971 0.0632393076839947 -0.07325757153283004 0.08309947084948825 -0.18587906751489314]] 
               [0.016781592856728436 -0.01693279279285234 -0.0002514603557369245])
          sample (sigmoid-sample-h-given-v hl [1 1 1 0 0 0])]
      (is (= 3 (count sample)))
      (is (every? #{0 1} sample)))))
