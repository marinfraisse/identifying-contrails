import os

RELOAD_MODEL = True
TEST_PLOT = False
TEST_API = True
TF_CHECKPOINT_PATH_all = os.path.join(os.path.expanduser('~'), "code", "marinfraisse", "identifying-contrails","idcontrails","tf_checkpoints","tf_checkpoint" )
TF_CHECKPOINT_PATH = os.path.join( os.path.dirname(__file__),"tf_checkpoints","tf_checkpoint")
# TEST_SAMPLES_PATH = os.path.join(os.path.expanduser('~'), "code", "marinfraisse", "identifying-contrails","tf_checkpoints" )


band_choice = ['band_11.npy','band_14.npy','band_15.npy']
target_suffix = 'human_pixel_masks.npy'
N_TIMES_BEFORE = 4

DATASET_SAMPLE_PATH = os.path.join(os.path.expanduser('~'), "code", "marinfraisse", "identifying-contrails","dataset_sample")

FIG_SAVES_PATH = os.path.join(os.path.expanduser('~'), "code", "marinfraisse", "identifying-contrails","plot_png_tests")


# ===================== WIP =================================


#getting the dataset directory
BASE_DIR = '/kaggle/input/google-research-identify-contrails-reduce-global-warming/train'
UPDATE_ID_LIST = False

IMG_SIZE_TARGET= 256
NUMBER_CHANNELS_TARGET= 3











# Defining training parameters
CHUNCK_SIZE = 1000
# NB_CHUNCKS = int(len(contrail_record_ids)/CHUNCK_SIZE) + 1
NB_CHUNCKS = 10
MODEL_NAME = "Unet64"

# Model Parameters
START_NEURONS = 64             # Number of start neurons
DROPOUT_RATIO = 1           # Dropout ratio vs classical Unet

# Training Parameters
LOSS_FUNCTION = "dice_loss"  # Loss function
# TRAINING_SET_SIZE = len(contrail_record_ids)    # Training set size
VALIDATION_SPLIT = 0.3                         # Validation split ratio
BATCH_SIZE = 16                                # Batch size
EPOCHS = 20                                   # Max number of epochs per chunk

# Callbacks Parameters
ES_PATIENCE = 6    # Early stopping patience
MAX_LR = 1e-4      # Maximum learning rate
MIN_LR = 5e-6      # Minimum learning rate
LRP_PATIENCE = 4   # Learning rate patience

# Optimizer Parameters
BETA_1 = 0.9       # Beta 1
BETA_2 = 0.999     # Beta 2
EPSILON = 1e-07    # Epsilon

# Final Dictionary
MODEL_PARAMS = {
    "name": MODEL_NAME,
    "model_parameters": {
        "start_neurons": START_NEURONS,
        "dropout_ratio_vs_classic": DROPOUT_RATIO
    },
    "training_parameters": {
        "loss_function": LOSS_FUNCTION,
        # "training_set_size": TRAINING_SET_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    },
    "callbacks_parameters": {
        "es_patience": ES_PATIENCE,
        "max_lr": MAX_LR,
        "min_lr": MIN_LR,
        "lrp_patience": LRP_PATIENCE
    },
    "optimizer_parameters": {
        "beta_1": BETA_1,
        "beta_2": BETA_2,
        "epsilon": EPSILON
    }
}


# contrail_record_ids = ['7796349705919134058',
#  '1323454114859044848',
#  '6931092088014476366',
#  '6829913604516271041',
#  '4220501911772948889',
#  '3932905402383045381',
#  '4778515849814329275',
#  '1616047197720010576',
#  '3587939903817787650',
#  '3371635388520693259',
#  '827698124032739472',
#  '6450526114496045665',
#  '8427762426551047845',
#  '356753751301700551',
#  '2012269165321285150',
#  '8813810217936458460',
#  '8766746156465708669',
#  '769442964726879842',
#  '4043700155902625134',
#  '8367280214868696052',
#  '495617192605792162',
#  '6784660536610603771',
#  '4668743094973273475',
#  '1323908870123553828',
#  '2886211455814528441',
#  '2914043542277215762',
#  '5606459396526048571',
#  '4680871618077643498',
#  '8738368405182301048',
#  '4028122642072153941',
#  '7400153696204247475',
#  '247790728255089189',
#  '4694915034393444297',
#  '3836558990699323594',
#  '7760802812914805068',
#  '4906039827361044875',
#  '2829674553563934229',
#  '1310949240439380947',
#  '1161525793329847144',
#  '7444502640525260662',
#  '7425325086659830922',
#  '2735110830284997549',
#  '3736028211167490688',
#  '4613557518829880586',
#  '7858044946613237082',
#  '8151517362689079282',
#  '5855346699829716929',
#  '1729763564275633845',
#  '5962174846620165826',
#  '2346860569459574392',
#  '7861948822468045244',
#  '4130188662038274161',
#  '345849117882649540',
#  '5645968110844843449',
#  '2822864707578145429',
#  '160834797934848624',
#  '4254579045844631730',
#  '3193954941728077424',
#  '5817457388831729378',
#  '5726431897529551851',
#  '6691869049413678138',
#  '7035137390254793172',
#  '9201937158328805399',
#  '8913406628326348192',
#  '7252980571954504949',
#  '5896444220083753281',
#  '2287944912139537249',
#  '4987706127073067867',
#  '5931882434014676831',
#  '1135874315020511661',
#  '6392538146640178933',
#  '5715545566729321173',
#  '5776141700240456127',
#  '632779570204844891',
#  '2053244798708416633',
#  '6923618144868948184',
#  '4613273809287972595',
#  '656421177149425032',
#  '694689085888200169',
#  '6599741607856138580',
#  '558927735179826864',
#  '5326464410838162178',
#  '8988348266920466498',
#  '7775615968776916351',
#  '3665529520964272841',
#  '8444103334521659825',
#  '7492656306791024460',
#  '6176420442560150390',
#  '4173680462695149944',
#  '4023461381779345888',
#  '5077296740911904436',
#  '5587339505087823578',
#  '259418980490048622',
#  '2879005905867436065',
#  '8175183503194315383',
#  '4761286974028164861',
#  '5632303571390031689',
#  '4031097137710481047',
#  '6117306641236462231',
#  '1490059920504718587',
#  '2535418204077330451',
#  '5278702659879218270',
#  '4235482590896276935',
#  '6067770806956494443',
#  '293706710005567226',
#  '2880893805327476726',
#  '5660788480503528462',
#  '3334488859402361680',
#  '502975682041109447',
#  '8053429081410206065',
#  '2598966158272425920',
#  '7953664221528484879',
#  '2469390254164456360',
#  '3305156738245736839',
#  '4684957878517209855',
#  '408647552826817870',
#  '832656124511757397',
#  '4322756982681839069',
#  '299562043820720690',
#  '3564185318298678051',
#  '3688663808595695130',
#  '5952070204773223730',
#  '4428591093706068366',
#  '2890277092364732969',
#  '8164550192050756938',
#  '3358004476359726060',
#  '2115742158894823648',
#  '7843061635485747022',
#  '3282567688222810787',
#  '8776129061707699361',
#  '3516759889828185992',
#  '6911633167962594641',
#  '838476834255491661',
#  '8361546802530464526',
#  '343777687141934188',
#  '2440402818222599338',
#  '6536878367045472349',
#  '4435769012356201548',
#  '8808941832209228789',
#  '5449133719927847055',
#  '7805688451431465985',
#  '5877062917364190774',
#  '1640760292989415590',
#  '4902304506136970897',
#  '974109821052472528',
#  '2157670232469759149',
#  '5645582001600329224',
#  '4279920333826199387',
#  '5648155499210392464',
#  '820532479838065985',
#  '2110582111871684677',
#  '4218549767397196754',
#  '9042372718221296562',
#  '5964192050151583837',
#  '2146459876432679891',
#  '7406512596615152469',
#  '405570864609733396',
#  '8547959753348375498',
#  '3834888687198402370',
#  '2612312995524200460',
#  '408602553549803836',
#  '1678166082105364925',
#  '117609053685710473',
#  '9201639692106934662',
#  '9054763200859451444',
#  '5457762604462951386',
#  '8083849078744540647',
#  '7059653049516666878',
#  '3818767654091440792',
#  '7450526085592900265',
#  '2690804457373209428',
#  '478551734703674075',
#  '2417789290442348424',
#  '5083238894467746327',
#  '6842782911929884469',
#  '3533232573606341684',
#  '3877135196585091852',
#  '1440257913527662385',
#  '688777900403570691',
#  '4656853525870907922',
#  '743916034622209128',
#  '3540199565652144233',
#  '3665362548193653834',
#  '5109293096785886393',
#  '2749778648106226643',
#  '2044743456328772763',
#  '342115186851410489',
#  '1015004752231541787',
#  '8800461065376958623',
#  '8999659785665101943',
#  '6109582717215954392',
#  '5572740044508598053',
#  '8600170205711720843',
#  '1392559205029642662',
#  '3784405930062868830',
#  '1543055461027582581',
#  '1000823728928031783',
#  '6297319594140705356',
#  '6157295309664018157',
#  '113033259589307043',
#  '115510078955066856',
#  '4476729056579184740',
#  '3475360190125245457',
#  '6503163128283127520',
#  '3593379795261168825',
#  '960228233488462975',
#  '3279203374447182022',
#  '1041708881855568056',
#  '2011945266626263355',
#  '716633797015437782',
#  '706446834852158837',
#  '3887446010198054254',
#  '3344412860940213848',
#  '5590306093120925183',
#  '1139157174154611422',
#  '8511754025671586990',
#  '2408975585346139129',
#  '3623702887273511479',
#  '3667203036761891432',
#  '6406381071677302374',
#  '590933086287866260',
#  '7093955858163956351',
#  '6366076074130833124',
#  '5987520984322247251',
#  '8133631370800540769',
#  '7011194171411153976',
#  '3002011962939012144',
#  '4782594089585143960',
#  '4354931535694850247',
#  '2686966689315793335',
#  '3327122431456945646',
#  '6029629369028312146',
#  '3288528556553804146',
#  '8698313777615502624',
#  '9192283018562815941',
#  '217255253055395866',
#  '4523686626433852454',
#  '7405156208985833535',
#  '2720913918481288795',
#  '8845896981201044625',
#  '5017921813987699768',
#  '2210963309511056585',
#  '2330910415667440378',
#  '3377660449439229066',
#  '1842687531218871097',
#  '1276886412364195466',
#  '8633753254846681387',
#  '569429604928793075',
#  '3944016972418610459',
#  '2559262649097972277',
#  '7858242175200612152',
#  '2574373974981595797',
#  '2741112688626043937',
#  '1690630865214490760',
#  '4847827210345973241',
#  '8592674122948162430',
#  '4643129664574703323',
#  '4711431409679697765',
#  '9175880718388651258',
#  '6395298788400751248',
#  '8183551258440774565',
#  '988163370871338338',
#  '8318256261024552889',
#  '8802946860224058449',
#  '7138523739582029733',
#  '7278461789666079931',
#  '7201620672194563137',
#  '8761631188525327986',
#  '6393709696657234284',
#  '6274740825834502470',
#  '3657121162886855629',
#  '6423277800992246598',
#  '5718504398262493727',
#  '4143918631823764233',
#  '6796987921936775122',
#  '7069445402878653323',
#  '8626551892671582676',
#  '2892555492366545838',
#  '1048966600944720544',
#  '3705700767387841898',
#  '1082297381767357636',
#  '6745049865026863155',
#  '9027466261492520249',
#  '3041669555657290884',
#  '4471996495654483299',
#  '2866749988702739489',
#  '4192544917559913576',
#  '6851397008112518442',
#  '6145861444368490348',
#  '7543118751819921832',
#  '6501229276426332848',
#  '8662040979159862679',
#  '1578034660562008538',
#  '1506814752747023938',
#  '2506688989902996574',
#  '494790410808447771',
#  '3169598832327448168',
#  '6825146085075697778',
#  '7143770722829873088',
#  '5279859800100705057',
#  '1738369129342039561',
#  '2951123436286228129',
#  '6082927363810447674',
#  '4425389725740332128',
#  '7840314609849638335',
#  '6299496062567451898',
#  '3803165706302066784',
#  '38200052949169754',
#  '2646023435310536047',
#  '4145413063670836585',
#  '4452746089886101039',
#  '3721175348858593764',
#  '186525810207059483',
#  '1316275618769741237',
#  '5379869084879649440',
#  '5917298889435919322',
#  '4503042889061508934',
#  '3493975730865440135',
#  '8245419656547485655',
#  '3712729647971798985',
#  '6675572703410262649',
#  '362408466617561192',
#  '5617044773957847691',
#  '7589564644038165121',
#  '2302161387109200394',
#  '2189526425279110356',
#  '7344719507399188033',
#  '1029379057144333137',
#  '439856163392787329',
#  '6605823599063567247',
#  '4640555188886042384',
#  '7504045473044290834',
#  '8421464389650690824',
#  '1185356420099333444',
#  '1538379551980087334',
#  '889723723067308984',
#  '1414833449189360782',
#  '8516905394423639556',
#  '6270931785870586383',
#  '859490511825075746',
#  '732856384646679198',
#  '4732024108445090775',
#  '5692260410699111543',
#  '7979990827110378982',
#  '1732471488032865895',
#  '7691527551809249473',
#  '1563801141223010059',
#  '2502140240545473747',
#  '4977362108845569874',
#  '6739033699237225086',
#  '7248764848592890672',
#  '5996515471261356796',
#  '3844548488230611530',
#  '6215488474380222182',
#  '1159540769658533538',
#  '660399264765257184',
#  '3069028060956582585',
#  '301062519371231574',
#  '3067275891172837629',
#  '357823368609679022',
#  '1499617458696694152',
#  '2761362401241469944',
#  '598966771587334531',
#  '3209149426723587598',
#  '4289838095664278646',
#  '2511373668654386828',
#  '7757263543012202646',
#  '7702063800677188669',
#  '2443519898687725487',
#  '1941763710825027970',
#  '4083808235368888492',
#  '6228470962971944071',
#  '2178771514760913240',
#  '7181434378644008854',
#  '2995829288712516373',
#  '7067416622020016149',
#  '9155979547098748557',
#  '5450202053947598467',
#  '4150982342512091074',
#  '8418158111341606926',
#  '8452276476489362785',
#  '8769176631880414288',
#  '2136237941659039558',
#  '5369636914027990722',
#  '4368996274678044962',
#  '1927181867054490025',
#  '8369077010997947759',
#  '6685504945784892942',
#  '4862737152895867884',
#  '3441818734561660798',
#  '882517108999244647',
#  '5180186324728723225',
#  '929799133695385329',
#  '8291249651296791672',
#  '8148381080424611480',
#  '6652769053524212359',
#  '508433819882274889',
#  '4151389625523768880',
#  '5067595891081383338',
#  '581534278957887260',
#  '6876747574026891092',
#  '6035333686315407293',
#  '5929301504829645060',
#  '7839418306617342059',
#  '834491007547154050',
#  '4954952042013549460',
#  '4102158963892658377',
#  '1129706563887251376',
#  '6517288309020491837',
#  '7486272053073889132',
#  '7038379586210020690',
#  '9056723716115026099',
#  '3660107114744383184',
#  '7835429689902694741',
#  '8701345716727761759',
#  '2851732914735883951',
#  '2633144168847012577',
#  '4889422516414510170',
#  '5274593255145615890',
#  '5049514641889878224',
#  '2432051847362415551',
#  '4936981945672090137',
#  '5746965860688789260',
#  '7949176048993727390',
#  '2339801735634567972',
#  '729425368198389489',
#  '535160896956548183',
#  '4931364177068312131',
#  '3564468928339576682',
#  '8982115718625421632',
#  '1061457468908604445',
#  '8351321489276498138',
#  '2829755941892094788',
#  '5873026420687258046',
#  '5095607247539737471',
#  '2434452320421837226',
#  '336919654598015725',
#  '7311660486980535456',
#  '1123848757551794175',
#  '8752053441627676471',
#  '4278351689039988851',
#  '5931318714640707597',
#  '2523259673272790271',
#  '6649300931988371171',
#  '4738521286428263971',
#  '8028220136038643634',
#  '1208501122897970093',
#  '7585255488006220865',
#  '3653713767573322655',
#  '6593864453862934829',
#  '729679142787518342',
#  '1457724398784743664',
#  '5479177379445018560',
#  '7809894886288079553',
#  '971988229895400255',
#  '6125540311021882624',
#  '2396873897355698830',
#  '8641804539284705080',
#  '3041805776109925982',
#  '3218259825010582940',
#  '8732403045801451364',
#  '311827246675488056',
#  '4145907924144188926',
#  '1586218680184808079',
#  '6486295826617424049',
#  '9084965046749553274',
#  '2568838981017487849',
#  '4755267965181835987',
#  '7781915641960278381',
#  '2709335147541043529',
#  '7226287038430170044',
#  '8817229479860486345',
#  '4670220516865026932',
#  '5175000367311135242',
#  '3349306449889320959',
#  '2256468876647484349',
#  '166435108423931339',
#  '2853584072934941871',
#  '1526865577245990194',
#  '9098690403704607232',
#  '2085580965186069738',
#  '5743619166308382658',
#  '3380839725237277344',
#  '2789185662656134967',
#  '3835179890521050764',
#  '5532822426837495880',
#  '8297993596675079372',
#  '7754519734558208485',
#  '1948547835903034396',
#  '5323778009247916992',
#  '1975223055445762263',
#  '6724308754482229969',
#  '2319980131910305585',
#  '4326075164644617569',
#  '7398206360802506859',
#  '4882026603194932752',
#  '7322425354570741648',
#  '6628624060814805093',
#  '7546497376117977361',
#  '578410020390848648',
#  '147152050802033169',
#  '6821706665549294948',
#  '1282082351427715345',
#  '2250324206342360047',
#  '8197980965919990855',
#  '5816745847836376582',
#  '2905619542333795623',
#  '9193061742185459497',
#  '384419004243630033',
#  '5833914761497313831',
#  '2027426640333066780',
#  '355981598128353055',
#  '2139176566707350260',
#  '2272871838590838299',
#  '5662446532904584734',
#  '7737270919033524579',
#  '2399048013454406246',
#  '8467335785520151484',
#  '8488894544961306158',
#  '9052668895427991638',
#  '3277105526089835128',
#  '2858582759856332064',
#  '2786500989969314943',
#  '6416865323320591885',
#  '1927561045927607292',
#  '8443915190215904823',
#  '8393394304681136825',
#  '6673270199210097198',
#  '7611017655945844478',
#  '1725845354325125137',
#  '6499016510280036356',
#  '4728336024252406774',
#  '6866300780424584270',
#  '3099726801169258092',
#  '5400797065914544986',
#  '1405265666642715278',
#  '6752193716605374270',
#  '6735194909711800535',
#  '4667949279061898176',
#  '5571865039619258434',
#  '1324011194742701128',
#  '5846486896202871998',
#  '7756069085662016692',
#  '6742881888282733684',
#  '7881054401275063854',
#  '2502961914222227698',
#  '7274054103912341855',
#  '4074150133585316582',
#  '7872827018578472341',
#  '5456681308006358752',
#  '8046531594726020838',
#  '7449872487706899848',
#  '8647660818664419944',
#  '4728851539442005572',
#  '4384204969775732258',
#  '1470311886449908198',
#  '845826757098102878',
#  '6462090745794886877',
#  '261997947821523733',
#  '3662150889016435641',
#  '3895600096178780167',
#  '33763174908501845',
#  '7019065580730886598',
#  '972640141000843452',
#  '8520859669845955362',
#  '322183095601570979',
#  '7456550801727090290',
#  '4697249169446260755',
#  '6278882310668586047',
#  '2469461365597784475',
#  '5528450905460122242',
#  '929664762844071148',
#  '938394751916430149',
#  '6426991466267061545',
#  '7201501776008948189',
#  '1563328488480263587',
#  '8770228561978153697',
#  '6952556170185935791',
#  '4205387167554399032',
#  '3653850452429541928',
#  '9006488511699401048',
#  '3618183904891024065',
#  '1944255890651033434',
#  '7275368624465079387',
#  '6189366144135268178',
#  '7080435823040616832',
#  '3269652203758820447',
#  '1796924488030040331',
#  '7927247435594864835',
#  '8011088137787721081',
#  '5594951198949235824',
#  '7352504087339780847',
#  '6620971495732379749',
#  '3604554488656123850',
#  '1019747901459344689',
#  '8194701754656832928',
#  '2751599984921471659',
#  '5968174353802385953',
#  '391533412759612949',
#  '4276379397654445677',
#  '4851153889183861141',
#  '3402772693950098142',
#  '7364008635611514136',
#  '50988259952989482',
#  '2023028626469218776',
#  '1925092644548228590',
#  '3779249767856817577',
#  '9022628719736710087',
#  '226509675393083650',
#  '2260294646759391458',
#  '6592179960674655922',
#  '8519119750219390242',
#  '5671326122959005267',
#  '3378276456146495746',
#  '8199425322706742072',
#  '9185132640380022140',
#  '8161659224635057350',
#  '931986421868455838',
#  '1830734353996273621',
#  '4520877944732149485',
#  '8846804505935920224',
#  '8912018265990022848',
#  '8692189119038834293',
#  '6626689316753541791',
#  '591693941053400254',
#  '986726145602616059',
#  '7899264727389582971',
#  '3278792320791773982',
#  '473137156325276895',
#  '4088767233087161870',
#  '3705585960191972023',
#  '6754001639975964226',
#  '8905784318750003271',
#  '7217981061308985585',
#  '624263041138075333',
#  '170888959230435248',
#  '2732895879388837725',
#  '7655457953964187238',
#  '1138885116957060676',
#  '635545611997535649',
#  '4676966328650105430',
#  '3593703694337714590',
#  '4357186746121980435',
#  '7952111137280287798',
#  '7093569970238788178',
#  '7253392376447203093',
#  '5595955646110591237',
#  '919137315966527772',
#  '6215896210154100391',
#  '1490610301426361492',
#  '2961515851619075745',
#  '1742437835356960082',
#  '445929687034288262',
#  '2510195073204762442',
#  '2691584131703761429',
#  '5342577176406475552',
#  '4652508329067442145',
#  '9218270074029979190',
#  '7841088638548290287',
#  '8448102696316736346',
#  '1819941796548763992',
#  '7698255922646016640',
#  '489798852347217974',
#  '5243787367769638958',
#  '1483607432775014873',
#  '4280705820175472644',
#  '5885530381307357522',
#  '2133152069966015742',
#  '3272352983210113266',
#  '1998339060921678564',
#  '3528681364673393928',
#  '8428430855974435825',
#  '7859213053622405657',
#  '5526960218813004640',
#  '1721761266965182756',
#  '3253179517742371594',
#  '2316679366459286143',
#  '2459548867604579277',
#  '8486922314507150839',
#  '7990955547905609642',
#  '1178248797439474438',
#  '8808788846725516543',
#  '5403129506750867678',
#  '5599878444548641163',
#  '7120629012181382857',
#  '8478237702746540889',
#  '7326221875665982358',
#  '4483788739710491687',
#  '4136569808879888725',
#  '7405094593550508479',
#  '2232743103247734085',
#  '3547556206125561967',
#  '4567565774230415854',
#  '7794317460587277423',
#  '192341828693511531',
#  '5347235455210950346',
#  '8954544363650949222',
#  '5110685343686579461',
#  '6879535282426753625',
#  '2610261643311685448',
#  '10733623242774169',
#  '61368388528777396',
#  '7133912117180643454',
#  '3167830239008208257',
#  '3290118800796857251',
#  '3854309337538428371',
#  '1613396822534946167',
#  '3685557716177346182',
#  '3289507438514028857',
#  '2935947551074924161',
#  '9137688875605038439',
#  '347767725293715254',
#  '7477478193013662895',
#  '8038286710365731667',
#  '819811568279557932',
#  '5747161549935964139',
#  '7640558979184660490',
#  '8875509368616637274',
#  '7700407424461592496',
#  '9110213233993494455',
#  '1190697280077790326',
#  '1966656733586886981',
#  '7545647478524789187',
#  '4337902339540092031',
#  '2135158158143724235',
#  '7385071493451218055',
#  '4969040864189209841',
#  '7377011870275482390',
#  '1158973730279189625',
#  '5695561647587146752',
#  '5984007163412268747',
#  '6993572356866457999',
#  '8558915452964246891',
#  '8920237029193168589',
#  '7724728738655266456',
#  '3726745381724926731',
#  '6409234069346887423',
#  '569608756220763482',
#  '86326882006458690',
#  '4487595363319371888',
#  '2579932902850230245',
#  '1874573531230821430',
#  '5610670417977930297',
#  '2112014794923111080',
#  '7548801515188209585',
#  '5391927837398800620',
#  '3727988927537047897',
#  '3756138508975242733',
#  '8507054814437735788',
#  '1767876474998843192',
#  '5389857977683352194',
#  '1053261235359945194',
#  '2258901233906478144',
#  '7407996838623399934',
#  '6698473799711888751',
#  '4772827491968866334',
#  '5250895750034545198',
#  '9088848127784570051',
#  '3407767190037612257',
#  '4863053699275157174',
#  '1549184721261256993',
#  '8453723085124295498',
#  '7133550338861567628',
#  '4968993712625542075',
#  '8607113346572745355',
#  '4613208243141386969',
#  '3911152600944827265',
#  '7380647150260569422',
#  '4264584757996644649',
#  '9123594341774178190',
#  '5364643196056600147',
#  '3946794122818855233',
#  '1207938035779504356',
#  '855791118207380172',
#  '6140718402126197883',
#  '6177328697988303676',
#  '6208918929810021851',
#  '6145156585642964756',
#  '2483215875668320557',
#  '2182658357996130924',
#  '919496064519167077',
#  '7570345990921815686',
#  '5030115286534683929',
#  '6536962563800073656',
#  '4787008586442356852',
#  '5292700592566427934',
#  '1929998620585100857',
#  '473936396483116380',
#  '993524570541918528',
#  '3236016298373797442',
#  '5955012978412351680',
#  '2502458804727943490',
#  '6117733847408076371',
#  '7522205077989088367',
#  '1754986182735608131',
#  '5591706081715869149',
#  '6302502318275515482',
#  '7491674875496614261',
#  '3492033651049268996',
#  '8959667497390380278',
#  '1111066390974440788',
#  '9142253991067304231',
#  '7319427807849756063',
#  '2177018971198860623',
#  '6317812053717439175',
#  '3388629713382399388',
#  '6051365979145244129',
#  '2270339721609441690',
#  '1937619829920621475',
#  '2700339704173024261',
#  '1245410098037715298',
#  '883273518039845381',
#  '4839521730548608587',
#  '7523516642556835375',
#  '2672586045575947335',
#  '643139180184146452',
#  '7594498794212875770',
#  '4943321637871355895',
#  '481873927077161022',
#  '8680237452812866033',
#  '3422788133786466448',
#  '8926256458251546660',
#  '8592733206293583452',
#  '6010196555988121266',
#  '5021095571122033691',
#  '5857073736785750826',
#  '2940662393215374276',
#  '7444200999966050576',
#  '4381418532596269346',
#  '1314913608015247668',
#  '4638202321412871823',
#  '2182010486963238036',
#  '1309835208004147826',
#  '1788943332379668287',
#  '1387378646810838453',
#  '1818067434646514058',
#  '6088674670576968847',
#  '6255941996885532872',
#  '8324838531262992522',
#  '5935971367980724198',
#  '7544465845217286727',
#  '41018993672621438',
#  '6517777164155021785',
#  '4003631587183307388',
#  '7461817293186913054',
#  '546037467310934921',
#  '50064023624604763',
#  '1775261217777999543',
#  '2662846210596416852',
#  '2375626243762379736',
#  '6864447628659737343',
#  '2095279965412730563',
#  '1673603382898039983',
#  '3358908282637153079',
#  '5910823622448872878',
#  '6818048530829620339',
#  '3376911515344335028',
#  '938693354601637937',
#  '2537313213980971969',
#  '1129898628300902569',
#  '4503606934945396652',
#  '6042122964862156880',
#  '7518306418178016426',
#  '9023443438883376666',
#  '1116630795124527275',
#  '6994628286415929454',
#  '494610631379455726',
#  '7607174067873740737',
#  '1004946886927737572',
#  '2142207711447026392',
#  '1808817626992390925',
#  '6702340488284465684',
#  '4134048204300128634',
#  '3639490580327928339',
#  '8198164412895662298',
#  '4574891982366703063',
#  '9090783651194491983',
#  '8412433757752234643',
#  '5461137288532942273',
#  '6076039653894588248',
#  '193104800146016358',
#  '8883172217039390588',
#  '6358708393854780966',
#  '1732442941720899096',
#  '3586417540492116245',
#  '4434375809683616649',
#  '7908794662021806955',
#  '7605560126552872379',
#  '9100849546387432031',
#  '6161239784331315696',
#  '8835802229362713032',
#  '5019145532587655260',
#  '4011707554197536166',
#  '3955168693247466189',
#  '2795963405068716285',
#  '7248100563522645890',
#  '6761158370524065358',
#  '7792844986869703228',
#  '2550038018158254866',
#  '4279771161524302655',
#  '3589712845244007573',
#  '5723719552902800675',
#  '7312814840198782061',
#  '663005000183730773',
#  '2779564180946166361',
#  '3420538625843618472',
#  '1310764900265706474',
#  '4932443468660127271',
#  '8153052889216038936',
#  '3839334354191968647',
#  '8641797850444914071',
#  '130855695818513737',
#  '7444656843355261448',
#  '2817214037755297378',
#  '5977431598343402199',
#  '4512692278242718050',
#  '8778451507553394441',
#  '2672507946899140875',
#  '2038694767865864829',
#  '5954795025686284348',
#  '3164639814243425891',
#  '559450465664794624',
#  '4528205043739945780',
#  '2211204524974061741',
#  '415726771833434591',
#  '1029669746148696544',
#  '10534680566306659',
#  '1845636773449803664',
#  '5466838335724481043',
#  '1934616502133345426',
#  '4193653470060032950',
#  '9215747383165264059',
#  '6358181583618956893',
#  '1110469894434959091',
#  '7978583257351994245',
#  '2715235061483165830',
#  '8677586109327293769',
#  '6323514309651027237',
#  '5602214017403190439',
#  '8587890062889606562',
#  '2664412749848823127',
#  '4559062683957000399',
#  '5250436967444904867',
#  '4906924426479903940',
#  '6970805130867770570',
#  '8946472931472308147',
#  '1595008208127698499',
#  '860208781617715647',
#  '5272913856148741731',
#  '3918029156718432578',
#  '5870157607170259131',
#  '4182421386811766738',
#  '9166847610597492947',
#  '5578357809756671740',
#  '5283947926400890428',
#  '562402355178081820',
#  '2951616330316823076',
#  '772843743538931830',
#  '2690953213822072057',
#  '3494134861297424665',
#  '164753183269622555',
#  '8437345773354556985',
#  '2008476055865331832',
#  '4807533567538560261',
#  '4497539820163331926',
#  '139474040382920694',
#  '1324808284776417133',
#  '6678956086118105032',
#  '8924600200225347575',
#  '7508677836646404125',
#  '3362063189832193270',
#  '7366295279323372139',
#  '3739702264591060069',
#  '828222075271454498',
#  '3277147824574833279',
#  '7683533512015798868',
#  '8773830518685163855',
#  '6282314372738983128',
#  '5072424789853271822',
#  '864475936315629629',
#  '8668075508905454989',
#  '3262221303410507655',
#  '4533088712641777739',
#  '7285570252036343467',
#  '1591424068450076051',
#  '8820226440628544450',
#  '8959002919123541960',
#  '1683708425649563397',
#  '8533427414470533642',
#  '5692010216233575271',
#  '8385862921770423823',
#  '8438286845763647331',
#  '1091333583887620331',
#  '3608984930475520254',
#  '4549466446246046247',
#  '1901347445187424069',
#  '1670112312018996421',
#  '4553984338996929734',
#  '6096042651778515649',
#  '8785414405848918237']
