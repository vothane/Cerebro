(ns cerebro.DBN_test
    (:require [clojure.test :refer :all]
              [cerebro.DeepBeliefNet :refer :all]))

(def dbn
  {
   :sigmoid-layers [{:weights [[ 0.14839871643137212 -0.08501163823540675 0.051985421731801734 -0.14855205346676653 -0.044137597789180485 -0.0701731889478024] 
                               [-0.10252046344168927  0.05177738360494413 0.132389904383267    -0.11088185248031387 -0.07047144939818163   0.1342016154235016] 
                               [ 0.11659342725429114 -0.07565106509550569 0.036360063996785374 -0.08211466451905437  0.09155141306199341  -0.16083974594778425]] 
                     :bias [0 0 0]} 
                    {:weights [[ 0.1913826375359961  0.19959579863963633  -0.09572763885651328] 
                               [-0.04920529688188052 0.006828221921254152 -0.17304539780927708] 
                               [-0.1938654179145095  0.1287133626717507   -0.06540140494644048]] 
                     :bias [0 0 0]}]  
   :rbm-layers [{:N 6
                 :weights [[ 0.14839871643137212 -0.08501163823540675 0.051985421731801734 -0.14855205346676653 -0.044137597789180485 -0.0701731889478024] 
                     [-0.10252046344168927  0.05177738360494413 0.132389904383267    -0.11088185248031387 -0.07047144939818163   0.1342016154235016] 
                     [ 0.11659342725429114 -0.07565106509550569 0.036360063996785374 -0.08211466451905437  0.09155141306199341  -0.16083974594778425]] 
                 :hbias [0 0 0] 
                 :vbias [0 0 0 0 0 0]} 
                {:N 6 
                 :weights [[ 0.1913826375359961  0.19959579863963633  -0.09572763885651328] 
                     [-0.04920529688188052 0.006828221921254152 -0.17304539780927708] 
                     [-0.1938654179145095  0.1287133626717507   -0.06540140494644048]] 
                 :hbias [0 0 0] 
                 :vbias [0 0 0]}] 
   :log-layer {:weights [[0 0 0] [0 0 0]] :bias [0 0] :N 6}
  })

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

(def sigmoid-layers [{:weights [[3.830040562610957 2.2572438689428482 1.1291324614561007 -4.0898381940839315 -2.9943059764521114 -1.4455110000150806] 
                               [-0.23531638963814172 -0.7089522379361282 2.0373878851804412 0.5388205984585597 0.00909457901571344 -1.4625908870311244] 
                               [-2.9787234309856543 -2.5183288604959335 0.9839791764425729 2.7849715512720685 1.579416020412225 -1.4473204127308446]] 
                     :bias [-0.5889345615736271 0.21725896752865348 -0.5106834878093905]} 
                     {:weights [[2.2679914975465905 -0.032221378861459635 -2.3239115409314617] 
                                [0.7196361212711985 0.3531154380197722 -0.698710183627769] 
                                [-3.856656951090344 1.1713952667178291 3.7199477886071834]] 
                      :bias [0.026740498859828266 -0.09725689788213573 -0.9354588375656511]}])

(def log-layer {:weights [[1.362310331208894 0.4118738320958368 -2.1298562224417825] 
                          [-1.362310331208894 -0.4118738320958366 2.1298562224417825]] 
                :bias [0.17633688988848428 -0.17633688988848387]
                :N 6})

(deftest DBN-predict-test
  (testing "DBN should make accurate predictions with mocked sigmoid and log layers"
    (let [dbn (assoc dbn :sigmoid-layers sigmoid-layers :log-layer log-layer)]
      (is (= (predict dbn [1 1 0 0 0 0]) [0.9653090289791912 0.03469097102080875])))))

(deftest DBN-test
  (testing "DBN should make accurate predictions after training"
    (let [dbn (DBN-pretrain dbn training-features 1000 0.1 1)
          dbn (finetune dbn training-features training-predictors 10 0.1)]
      (is (= (predict dbn [1 1 0 0 0 0]) "This will always fail.")))))

(def sigmoid_layers [{:weights [[3.830040562610957 2.2572438689428482 1.1291324614561007 -4.0898381940839315 -2.9943059764521114 -1.4455110000150806] 
                                [-0.23531638963814172 -0.7089522379361282 2.0373878851804412 0.5388205984585597 0.00909457901571344 -1.4625908870311244] 
                                [-2.9787234309856543 -2.5183288604959335 0.9839791764425729 2.7849715512720685 1.579416020412225 -1.4473204127308446]] 
                      :bias [-0.5889345615736271 0.21725896752865348 -0.5106834878093905]} 
                     {:weights [[2.2679914975465905 -0.032221378861459635 -2.3239115409314617] 
                                [0.7196361212711985 0.3531154380197722 -0.698710183627769] 
                                [-3.856656951090344 1.1713952667178291 3.7199477886071834]] 
                      :bias [0.026740498859828266 -0.09725689788213573 -0.9354588375656511]}])

(def init_log_layer {:weights [[0 0 0] [0 0 0]] :bias [0 0] :N 6})

(def log_layer {:weights [[0.06358695635437592 0.03800054523717951 -0.06642946580979513] 
                          [-0.06358695635437592 -0.0380005452371795 0.06642946580979511]] 
                :bias [-0.0028405655699967128 0.0028405655699967145] 
                :N 6})
              
(deftest DBN-finetune-test
  (testing "DBN should finetune correctly"
    (let [dbn (assoc dbn :sigmoid-layers sigmoid_layers :log-layer init_log_layer)
          X (transient [[0 0 1] [0 1 1] [0 0 1] [1 1 0] [1 1 0] [0 1 0] [0 0 1] [0 0 1] [0 0 1] [1 1 0] [1 1 1] [1 1 0] [0 0 1] [0 1 1] [0 1 1] [1 1 0] [1 0 0] [1 1 0]])]
      (with-redefs-fn {#'sample-hidden-layers (fn [a b]
                                                (let [x (nth X (dec (count X)))
                                                      _ (when-not (= 0 (count X)) (pop! X))]
                                                  x))}   
        #(is (= (:log-layer (finetune dbn training-features training-predictors 3 0.1)) log_layer))))))
        
