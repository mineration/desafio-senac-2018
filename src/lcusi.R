library(data.table)
library(dplyr)
library(stringr)
library(keras)
keras::use_condaenv('r-tensorflow', required = T)
library(rsample)
library(tokenizers)
library(stopwords)
library(readr)


# preguica de tratar titulo e texto separado, vou colar os dois logo
# pto_atencao4 se der tempo, avaliar impacto de subredes pra titulo e texto 
trei <- fread("desafio-senac-2018-master/data/avaliacoes_train.csv", encoding = 'UTF-8') %>% 
  as_data_frame() %>% 
  glimpse() %>% 
  mutate(tx = paste(Titulo, Texto, sep='ª') %>% 
           str_remove_all('\t|\n'))
test <- fread("desafio-senac-2018-master/data/avaliacoes_test.csv", encoding = 'UTF-8') %>% 
  as_data_frame() %>% 
  glimpse() %>% 
  mutate(tx = paste(Titulo, Texto, sep='ª') %>% 
           str_remove_all('\t|\n'))

dic.sb <- readLines('sb_dic', encoding = 'UTF-8')  # a parte de simbolos do dicionario
writeLines(paste(sort(unique(strsplit(dic.sb, '')[[1]])), collapse=''), 'dic_sb')
dic.sb <- readLines('dic_sb')
# dic <- sort(unique(c(letters,  # base da char-level convnet (zhang, zhao, lecun)
#                      0:9,
#                      # ' ',  # nao inclui espaco pq keras::text_tokenizer ja trata isso 
#                      strsplit(dic.sb, '')[[1]])))

troca.sb <- function(tx, ls.sb = dic.sb, troca.por = ' '){
  Reduce(function(string, sb) str_replace_all(string = string,
                                              pattern = fixed(sb),
                                              replacement = troca.por), 
         str_split(ls.sb, '')[[1]], 
         tx)
}

tx.test.stem <- paste(test$Titulo, test$Texto) %>%
  troca.sb() %>% 
  str_to_lower() %>% 
  str_replace_all('comida', 'comidaa') %>%  # comida tava virando "com" nesse stemer de pobre
  str_replace_all('bras.lia', 'brasiliaa') %>%  # brasilia tava virando "brasil" tbm
  str_remove_all('(uns)|(umas)|(ah+)') %>%  # 'uns' 'umas' 'ahhh' nao incluidos em stopwords('pt')
  tokenize_word_stems(language = 'portuguese', 
                      stopwords = stopwords::stopwords('pt')) %>% 
  lapply(paste, collapse = ' ')
tx.test.stem.tok <- tx.test.stem %>% 
  tokenize_skip_ngrams(lowercase = TRUE, 
                       n_min = 1,
                       n = 3,
                       k = 1) %>% 
  lapply(str_replace_all, pattern = ' ', replacement = '_') %>% 
  sapply(paste, collapse = ' ')

tam.dic <- 30000L  # 30 mil ja ta ok, segundo alguns influenciadores digitais

tok <- text_tokenizer(num_words = tam.dic, filters = '')  # ja filtrei antes, com mais controle
fit_text_tokenizer(tok, tx.test.stem.tok)
seq.tx.test.stem.tok <- texts_to_sequences(tok, tx.test.stem.tok)
# hist(sapply(seq.tx.test.stem.tok, length))

corte.tam <- 100L  # considero suficiente 100 ngrams

seq.tx.test.stem.tok <- seq.tx.test.stem.tok %>% 
  pad_sequences(maxlen = corte.tam,  # quero ate 100 ngrams
                padding = 'post', 
                truncating = 'post')

tx.trei.stem <- paste(trei$Titulo, trei$Texto) %>%
  troca.sb() %>% 
  str_to_lower() %>% 
  str_replace_all('comida', 'comidaa') %>%
  str_replace_all('bras.lia', 'brasiliaa') %>%
  str_remove_all('(uns)|(umas)|(ah+)') %>% 
  tokenize_word_stems(language = 'portuguese', 
                      stopwords = stopwords::stopwords('pt')) %>% 
  lapply(paste, collapse = ' ')

tx.trei.stem.tok <- tx.trei.stem %>% 
  tokenize_skip_ngrams(lowercase = TRUE, 
                       n_min = 1,
                       n = 3,
                       k = 1) %>% 
  # (bi)(tri)grams sao palavras (incluindo emojis) separadas por sublinhado
  lapply(str_replace_all, pattern = ' ', replacement = '_') %>% 
  sapply(paste, collapse = ' ')

seq.tx.trei.stem.tok <- texts_to_sequences(tok, tx.trei.stem.tok)
# hist(sapply(seq.tx.trei.stem.tok, length))
seq.tx.trei.stem.tok <- seq.tx.trei.stem.tok %>% 
  pad_sequences(maxlen = corte.tam, 
                padding = 'post', 
                truncating = 'post')  # post pad mais bonito de ler do q pre
# dim(seq.tx.trei.stem.tok)
# dim(seq.tx.test.stem.tok)
# tok$index_word[seq.tx.trei.stem.tok[1, seq.tx.trei.stem.tok[1, ] > 0]]

# saveRDS(seq.tx.trei.stem.tok, 'matriz.trei.rds')
# saveRDS(seq.tx.test.stem.tok, 'matriz.test.rds')

# agora o script dos experimentos de validacao cruzada (CV) 3-folds

ncv <- 3


# seq.tx.trei.stem.tok <- readRDS('matriz.trei.rds')
set.seed(31867)
ams <- rsample::vfold_cv(trei, ncv, 1, 'Avaliacao')  # seria perigoso nao estratificar
ic.ams <- lapply(ams[['splits']], '[[', 'in_id')


roda.fasttext <- function(embedding_dims = 50,
                          dropout1 = .4,
                          dropout2 = .3,
                          optim = 1L,
                          lr.mult = 0,
                          decay = 0.0001){
  max.epoch = 100L
  batch_size = 420  # nao deu tempo de testar o impacto disso          
  
  FLAGS <- list()
  FLAGS$embedding_dims <- embedding_dims
  FLAGS$dropout1 <- dropout1
  FLAGS$dropout2 <- dropout2
  FLAGS$optim <- optim
  FLAGS$lr.mult <- lr.mult
  FLAGS$decay <- decay
  
  cat(dropout1,
      dropout2,
      optim,
      lr.mult,
      decay,
      '\n',
      sep=', ')
  
  
  optimizer <- switch(c('sgd', 'adam', 'rmsprop')[FLAGS$optim],
                      sgd = optimizer_sgd(lr = .01 * exp(FLAGS$lr.mult),
                                          decay = min(exp(FLAGS$decay), 
                                                      .01 * exp(FLAGS$lr.mult) / max.epoch)),
                      adam = optimizer_adam(lr = .001 * exp(FLAGS$lr.mult),
                                            decay = min(exp(FLAGS$decay), 
                                                        .001 * exp(FLAGS$lr.mult) / max.epoch)),
                      rmsprop = optimizer_rmsprop(lr = .001 * exp(FLAGS$lr.mult),
                                                  decay = min(exp(FLAGS$decay), 
                                                              .001 * exp(FLAGS$lr.mult) / max.epoch)))
  
  input.shape <- c(corte.tam, tam.dic)
  
  medidas <- rep(NA_real_, ncv)
  
  for(ic.valid in 1:ncv){gc(); gc()
    mod <- keras_model_sequential()
    mod %>%
      layer_embedding(
        input_dim = tam.dic, 
        output_dim = FLAGS$embedding_dims, 
        input_length = ncol(seq.tx.trei.stem.tok)
      ) %>%
      layer_dropout(FLAGS$dropout1) %>% 
      layer_global_average_pooling_1d() %>%
      layer_dropout(FLAGS$dropout2) %>% 
      layer_dense(1, activation = "sigmoid")
    mod %>% compile(
      optimizer = optimizer,
      loss = 'binary_crossentropy',
      metrics = 'accuracy'
    )
    pasta <- sprintf('./FIMfasttext_%s/rodada.cv_%d',
                     do.call(sprintf,
                             c(paste(rep('%.3f', length(FLAGS)),
                                     collapse='_'),
                               FLAGS)),
                     ic.valid)
    print(pasta)
    if (!dir.exists(pasta)) dir.create(pasta, recursive = T)
    hst.fit <- try(fit(mod,
                       x = seq.tx.trei.stem.tok[ic.ams[[ic.valid]], ],
                       y = trei$Avaliacao[ic.ams[[ic.valid]]],
                       validation_data = list(seq.tx.trei.stem.tok[-ic.ams[[ic.valid]], ],
                                              trei$Avaliacao[-ic.ams[[ic.valid]]]),
                       epochs = max.epoch,
                       batch_size = 315,
                       verbose = 0,
                       callbacks = list(
                         callback_early_stopping(patience = 2),
                         callback_csv_logger(paste0(pasta, '/log.csv')),
                         callback_model_checkpoint(paste0(pasta, '/pesos.hdf5')))))
    if(class(hst.fit) == 'try-error') {
      medidas[ic.valid] <- rnorm(1, 3, .01)
    } else {
      saveRDS(hst.fit, paste0(pasta, '/hst.fit.rds'))
      medidas[ic.valid] <- min(hst.fit$metrics$val_loss)
    }
  }
  return(list(Score = -max(medidas), Pred = 0))  # retorna pior logloss da rodada de CV
}
# ls.hprm <- list.files(pattern = 'bayesOpt_fasttext_[0-9]+.csv')
# hst.hprm <- read_csv(tail(ls.hprm[order(as.numeric(gsub('[^0-9]', '', ls.hprm)))], 1))
# dput(hst.hprm)

# Precisei de alguns (95) experimentos ate chegar a hiperparametros (hprm) otimos -> hst.hprm
#       - A coluna hst.hprm$Value abaixo representa o retorno da funcao roda.fasttext pra
#   cada conjunto de hprm testado, i.e., o logloss do pior modelo dentre os tres ajustados
#    na rodada de CV. Value eh -logloss (negativa) pq meu algoritmo busca maximizar
hst.hprm <- structure(list(embedding_dims = c(37L, 17L, 30L, 86L, 65L, 46L, 
                                                 62L, 11L, 28L, 96L, 38L, 65L, 10L, 10L, 66L, 84L, 45L, 56L, 23L, 
                                                 45L, 25L, 21L, 39L, 46L, 71L, 42L, 100L, 100L, 80L, 57L, 70L, 
                                                 33L, 43L, 10L, 100L, 10L, 16L, 84L, 78L, 40L, 90L, 100L, 38L, 
                                                 70L, 54L, 69L, 54L, 100L, 10L, 10L, 72L, 53L, 73L, 88L, 73L, 
                                                 29L, 43L, 100L, 65L, 58L, 21L, 29L, 100L, 83L, 66L, 22L, 28L, 
                                                 35L, 64L, 31L, 24L, 69L, 82L, 50L, 61L, 87L, 71L, 83L, 95L, 99L, 
                                                 71L, 64L, 100L, 14L, 76L, 44L, 46L, 58L, 100L, 51L, 60L, 10L, 
                                                 100L, 40L, 95L), dropout1 = c(0.0259775001322851, 0.341277082334273, 
                                                                               0.177354817278683, 0.345716386795671, 0.411378282169546, 0.435043497355085, 
                                                                               0.0241770797410254, 0.068159934511512, 0.0481764766154811, 0.270354713167089, 
                                                                               0.7, 0.577428839017264, 0.0399430621405382, 0.7, 0.7, 0.428011806681752, 
                                                                               0.126742687844671, 0.37470771586959, 0.350783420434708, 0.384745448157951, 
                                                                               0.640436322322105, 0.442319609876722, 0.139353934830583, 0.108993934215144, 
                                                                               0.566282553821839, 0.262599591444942, 0.0813477436570536, 0.0690139205061352, 
                                                                               0.0577873219288493, 0.123691454596231, 0.603558709211697, 0.309388950433903, 
                                                                               0.666493337969363, 0.101553718041319, 0.432793409443444, 0.657538218043205, 
                                                                               0.198481956558573, 0.152486445386869, 0.240782919506718, 0.529238758992908, 
                                                                               0.188467257922195, 0.385018806613417, 0.242366342386231, 0.105909748049453, 
                                                                               0.113375178983298, 0.272542844838603, 0.286513767438896, 0.367838450404026, 
                                                                               0.697436212659111, 0.409863016844485, 0.179615063103847, 0.164133160017305, 
                                                                               0.555065400843369, 0.108072471315973, 0.261222200832167, 0.7, 
                                                                               0.45134837286963, 2.22044604925031e-16, 0.215276583268874, 2.22044604925031e-16, 
                                                                               2.22044604925031e-16, 0.303174971197749, 2.22044604925031e-16, 
                                                                               0.680597813481289, 0.0204827356274675, 0.00298508386632858, 0.398503086204753, 
                                                                               0.00746771018495933, 0.282428225895419, 0.414563928660937, 0.633516747080357, 
                                                                               0.342332263686694, 2.22044604925031e-16, 0.212627897371412, 0.161962787270054, 
                                                                               0.0792142644888885, 0.650399097939931, 0.274408234485205, 0.411749835288144, 
                                                                               0.534868474015879, 0.683067483789298, 0.377908969996497, 0.50897940436178, 
                                                                               0.317515002351332, 0.154273635338273, 0.0567083007404137, 0.476112774864998, 
                                                                               0.0350049932418715, 0.7, 0.136501467224605, 0.654516753479251, 
                                                                               0.100316685940426, 0.596905303572606, 0.000584965106334733, 0.230398691281989
                                                 ), dropout2 = c(0.62254007931333, 0.333277235459536, 0.42805465743877, 
                                                                 2.22044604925031e-16, 0.505718984189818, 0.7, 0.207864820638326, 
                                                                 0.418538256267553, 0.552755395509303, 0.64441714376246, 2.22044604925031e-16, 
                                                                 2.22044604925031e-16, 2.22044604925031e-16, 0.7, 2.22044604925031e-16, 
                                                                 0.208801626600325, 0.124094280344434, 0.192355502405155, 0.7, 
                                                                 0.0548698911437033, 0.203292108625551, 0.361369261983782, 0.696698315741222, 
                                                                 0.373434383956427, 0.7, 0.602772245467669, 2.22044604925031e-16, 
                                                                 2.22044604925031e-16, 0.194508067139541, 0.45353061737575, 0.7, 
                                                                 2.22044604925031e-16, 0.338578766640158, 0.7, 2.22044604925031e-16, 
                                                                 2.22044604925031e-16, 0.55969185799109, 2.22044604925031e-16, 
                                                                 0.7, 0.7, 0.268652561299202, 0.447099421318295, 0.058313670870848, 
                                                                 0.686005745083094, 0.7, 0.247930174459346, 0.453646618560618, 
                                                                 0.438743912939592, 2.22044604925031e-16, 0.231774380148366, 0.0725190542405471, 
                                                                 2.22044604925031e-16, 2.22044604925031e-16, 0.220830607670359, 
                                                                 0.054414451392069, 0.7, 0.411076418077093, 0.7, 0.697918309189084, 
                                                                 0.7, 2.22044604925031e-16, 0.209606204579413, 2.22044604925031e-16, 
                                                                 0.699397569477622, 0.411332576625973, 0.168475076984174, 0.371950618516786, 
                                                                 0.339321718085554, 0.157992530751151, 0.0995449384208769, 0.186211846328426, 
                                                                 0.25627042145934, 0.7, 0.7, 0.194962926458658, 0.641895832158689, 
                                                                 0.391467845719364, 0.0716524517668994, 0.223619743455173, 0.498962906421165, 
                                                                 0.675899905280657, 0.524885981273837, 0.12132915236232, 0.287710605328795, 
                                                                 0.61961942811968, 0.568907114759684, 0.604845227387259, 0.204608919835428, 
                                                                 0.564588241934808, 0.37989445287832, 0.620279531578529, 0.30136646935335, 
                                                                 0.5485533180352, 0.7, 0.268467905136809), optim = c(2L, 3L, 1L, 
                                                                                                                     2L, 1L, 2L, 2L, 3L, 2L, 2L, 3L, 3L, 1L, 3L, 1L, 2L, 1L, 2L, 2L, 
                                                                                                                     1L, 3L, 3L, 2L, 1L, 3L, 2L, 3L, 1L, 3L, 3L, 2L, 2L, 2L, 1L, 2L, 
                                                                                                                     3L, 2L, 3L, 1L, 3L, 1L, 1L, 1L, 2L, 1L, 2L, 1L, 3L, 2L, 2L, 3L, 
                                                                                                                     3L, 2L, 2L, 2L, 1L, 2L, 3L, 1L, 3L, 3L, 1L, 2L, 2L, 2L, 3L, 3L, 
                                                                                                                     3L, 3L, 3L, 2L, 2L, 3L, 2L, 3L, 1L, 3L, 2L, 1L, 2L, 2L, 2L, 1L, 
                                                                                                                     2L, 3L, 1L, 3L, 1L, 3L, 2L, 2L, 2L, 2L, 1L, 3L), lr.mult = c(1.5865325788036, 
                                                                                                                                                                                  -1.9825382605195, 2.09320325939916, -2.81107976026068, -2.43682950545994, 
                                                                                                                                                                                  2.76927835247623, -3.87606305371444, -1.85659594088993, -1.47422781144269, 
                                                                                                                                                                                  -2.31776011485789, 1.24032393149958, 4, -5, 3.3649893021204, 
                                                                                                                                                                                  3.28838746650758, 2.22835259884596, -1.16919525293633, -4.23597265323227, 
                                                                                                                                                                                  0.0914621235062878, -2.00570159960157, -4.61756246625093, -2.79779644706286, 
                                                                                                                                                                                  3.23463185850679, -3.47968899796286, -4.96283026553449, -4.61807714009514, 
                                                                                                                                                                                  4, 4, -5, 4, 4, 4, -3.19902303136175, 4, -5, 4, -0.753359292021383, 
                                                                                                                                                                                  4, -5, 4, -5, 4, 0.197880269261077, -1.83711419347674, 4, -1.65543981291667, 
                                                                                                                                                                                  3.18035948150837, 4, -5, 4, -1.61439022049308, -5, 4, 1.58676994382138, 
                                                                                                                                                                                  -0.803721445648325, 4, 1.89374326567858, -5, -1.62267954304789, 
                                                                                                                                                                                  4, 4, 3.10968398624635, 4, -1.0420305717146, 0.184606112570251, 
                                                                                                                                                                                  1.6734194560644, 3.8882097908735, 3.14693209111249, 0.147529081964003, 
                                                                                                                                                                                  -2.4073094134219, 0.857058755382813, 0.54866071883589, 4, -1.57827222237082, 
                                                                                                                                                                                  3.24600040438388, 0.255112960052335, 3.81640817006731, -5, 3.14223427883815, 
                                                                                                                                                                                  -1.60960984243089, 1.74755110137987, -3.1767184282653, -5, 1.91983679663817, 
                                                                                                                                                                                  2.79636615023405, 4, 3.9530149730749, -5, 4, 1.2064524629895, 
                                                                                                                                                                                  3.81890087257675, 4, 1.41440402059883, -5, 1.07817899709043), 
                              decay = c(-10.2810208564624, -10.4674540963024, -10.8428575275466, 
                                        -7.15659263835259, -7.75572959354654, -9.97105025866177, 
                                        -10.8383540639201, -8.9227750253712, -10.0223761200905, -10.9705962935818, 
                                        -7, -7, -7, -10.809354317435, -8.48943238133296, -7.89223161153495, 
                                        -9.38688612729311, -9.98057247701131, -9.23737109545626, 
                                        -10.6891081636898, -10.5926631110702, -9.62500858400017, 
                                        -10.8110711123234, -10.8872819124545, -7.34068225568805, 
                                        -10.0301444281849, -7, -10.298866066829, -8.85574292303723, 
                                        -8.40590237615079, -10.4966606729522, -9.24190019450599, 
                                        -8.59469625731547, -10.9690888946098, -8.06465487033715, 
                                        -8.64341762801712, -10.2396511331925, -10.8257211836427, 
                                        -10.4598596074203, -7.3631697282674, -10.9880441979052, -9.82031236863181, 
                                        -9.15306595619768, -9.22075848001987, -10.1636968826662, 
                                        -8.92863440741381, -9.00094899010009, -9.7730361650987, -10.883137700454, 
                                        -7.11453403696928, -8.44547176454216, -9.2594594566067, -7.58481426365595, 
                                        -8.50073751993476, -8.12728065479379, -10.5782931377124, 
                                        -8.02001207111847, -7, -7.15191648817478, -9.31573989297758, 
                                        -7.85175071060659, -10.8275629082679, -9.13462708160226, 
                                        -8.36232912981816, -7.82371143358885, -8.08240261494872, 
                                        -8.29438849188153, -8.48916013455818, -7, -10.9050185242668, 
                                        -7.95779070109655, -8.64661391731352, -9.34259140439511, 
                                        -10.1527467365157, -10.8612710875382, -10.2307338319607, 
                                        -9.33201840226985, -7.08913069328494, -7.237211715477, -8.29277580296166, 
                                        -8.92286611573037, -10.0027791084722, -7, -7.94792733912292, 
                                        -7.52135468530706, -7, -9.9251303708608, -7, -7, -7.08491354708993, 
                                        -7, -9.78058416821655, -7, -8.20178342862272, -9.13748371285222
                              ), Value = c(-0.211136104498544, -0.318843248907639, -0.40942283735975, 
                                           -0.30060125871197, -0.433946164891772, -0.214670878409853, 
                                           -0.400374761094218, -0.320877195700355, -0.226406095104049, 
                                           -2.99512047612066, -0.219158070526369, -0.21479627425256, 
                                           -3.00409137011484, -0.237984446646727, -3.01198349254587, 
                                           -0.210914051119724, -0.421020225176345, -0.441794880382393, 
                                           -0.213926236064214, -0.425072929619447, -0.594920195963072, 
                                           -0.387723441195229, -0.215023715393213, -0.499868905414706, 
                                           -0.569593874656636, -0.533376075651335, -0.23621270676022, 
                                           -0.406253628108812, -0.550330569238766, -0.222871626281868, 
                                           -0.213359318714103, -0.212656284763437, -0.384606737805449, 
                                           -0.411891194303399, -0.524272932954457, -0.222297534022642, 
                                           -0.214183135727501, -0.240328727087573, -0.633988640230635, 
                                           -0.222707138882707, -0.630531162023544, -0.407941192064596, 
                                           -0.418536356933739, -0.227299062130244, -0.409022697288057, 
                                           -0.213651305752928, -0.40999502559071, -0.218900311414314, 
                                           -0.636136115245197, -0.215987223605423, -0.216077748544352, 
                                           -0.575541730808175, -0.213502496987095, -0.212619590127598, 
                                           -0.212195725949562, -0.410209438237159, -0.210856888888647, 
                                           -0.542079159746999, -0.422496994066498, -0.220176414060204, 
                                           -0.24525450344157, -0.409448843287385, -0.216769885636218, 
                                           -0.212481952110386, -0.212274335648703, -0.216841791452759, 
                                           -0.226977975026745, -0.218223372514805, -0.21764597831213, 
                                           -0.32723571975594, -0.210704709200755, -0.210975051452608, 
                                           -0.217233922292033, -0.224251492880285, -0.239961588350327, 
                                           -0.416564436062523, -0.224852195839681, -0.545672077523625, 
                                           -0.41117767698091, -0.213121982615279, -0.211561816258599, 
                                           -0.370704686998025, -0.629999426395997, -0.212120609641399, 
                                           -0.217626315177135, -0.410492761627487, -0.220097372794281, 
                                           -0.628742781022321, -0.224760964026918, -0.212307058699915, 
                                           -0.211229835636914, -0.215628434175059, -0.209506233647952, 
                                           -0.621055769531623, -0.219551200937966)), row.names = c(NA, 
                                                                                                   -95L), class = c("tbl_df", "tbl", "data.frame"), spec = structure(list(
                                                                                                     cols = list(embedding_dims = structure(list(), class = c("collector_integer", 
                                                                                                                                                              "collector")), dropout1 = structure(list(), class = c("collector_double", 
                                                                                                                                                                                                                    "collector")), dropout2 = structure(list(), class = c("collector_double", 
                                                                                                                                                                                                                                                                          "collector")), optim = structure(list(), class = c("collector_integer", 
                                                                                                                                                                                                                                                                                                                             "collector")), lr.mult = structure(list(), class = c("collector_double", 
                                                                                                                                                                                                                                                                                                                                                                                  "collector")), decay = structure(list(), class = c("collector_double", 
                                                                                                                                                                                                                                                                                                                                                                                                                                     "collector")), Value = structure(list(), class = c("collector_double", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "collector"))), default = structure(list(), class = c("collector_guess", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              "collector"))), class = "col_spec"))

hprm.fim <- hst.hprm %>% 
  arrange(desc(Value)) %>% 
  slice(1) %>% 
  select(-Value)

vl <- do.call(roda.fasttext, hprm.fim)

pasta.fasttext <- function(embedding_dims = 50,
                           dropout1 = .4,
                           dropout2 = .3,
                           optim = 1L,
                           lr.mult = 0,
                           decay = 0.0001){
  FLAGS <- list()
  FLAGS$embedding_dims <- embedding_dims
  FLAGS$dropout1 <- dropout1
  FLAGS$dropout2 <- dropout2
  FLAGS$optim <- optim
  FLAGS$lr.mult <- lr.mult
  FLAGS$decay <- decay
  pasta <- sprintf('./FIMfasttext_%s/rodada.cv_%d',
                   do.call(sprintf,
                           c(paste(rep('%.3f', length(FLAGS)),
                                   collapse='_'),
                             FLAGS)),
                   1:ncv)
  pasta
}
pastas <- do.call(pasta.fasttext, hprm.fim)
mod.fim <- lapply(pastas, list.files, pattern = 'hdf5', full.names = TRUE) %>% 
  lapply(load_model_hdf5)

# prefiro combinar o resultado dos tres modelos, ja que sao leves
pred.cv.test <- lapply(mod.fim, predict, x = seq.tx.test.stem.tok)
pred.test.fim <- Reduce("+", pred.cv.test) / length(pred.cv.test)

avlc.test <- fread('desafio-senac-2018-master/data/avaliacoes_test.csv', encoding = 'UTF-8') %>% 
  as_data_frame() %>% 
  mutate(pred = pred.test.fim[, 1]) %>% 
  arrange(pred)
avlc.test %>%  
  head(20) %>%
  transmute(tx = paste(Titulo, Texto, sep='\n')) %>% 
  pull(tx) %>% 
  cat(sep = '\n\n')  # ok
avlc.test %>%  
  tail(20) %>%
  transmute(tx = paste(Titulo, Texto, sep='\n')) %>% 
  pull(tx) %>% 
  cat(sep = '\n\n')  # ok

avlc.test %>% 
  mutate(pred.arredondada = round(pred, 3)) %>%
  select(ID, pred.arredondada) %>%
  # fwrite('submissao2 - global_max_polling.csv', col.names = F) %>%
  fwrite('submissao2 - global_max_polling arredondada.csv', col.names = F)
