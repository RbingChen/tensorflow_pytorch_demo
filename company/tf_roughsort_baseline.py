#! /usr/bin/env python
# coding: utf-8
import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(sys.path[0]), 'mlx_release', 'python'))  # noqa
import tensorflow as tf
import mlx.python.tf as tfmlx
import math
from custom_layer import *
from mlx.python.util.graph_state_helper import graph_state_helper

# 特征
conf = {
    "dense": {
        "user_dense_len": 128,
        "item_dense_len": 107,
        "user_item_dense_len": 12
    },
    "sparse": {
        "user_sparse_list": [
            "1  f_user_1h_most_view_item_screening  8",
            "2  f_user_1h_most_view_cate_screening  8",
            "3  f_user_1h_most_view_type_screening  8",
            "4  f_user_1h_most_view_class_screening 8",
            "5  f_user_1h_most_two_view_cate_screening  8",
            "6  f_user_1h_most_three_view_class_screening   8",
            "7  f_user_6h_most_view_type_screening  8",
            "8  f_user_6h_most_view_item_screening  8",
            "9  f_user_6h_most_two_view_cate_screening  8",
            "10 f_user_6h_most_three_view_class_list_screening  8",
            "11 f_user_1d_most_view_item_screening  8",
            "12 f_user_1d_most_view_type_screening  8",
            "13 f_user_1d_most_two_view_cate_screening  8",
            "14 f_user_1d_most_three_view_class_screening   8",
            "15 f_user_3d_most_view_item_screening  8",
            "16 f_user_3d_most_view_type_screening  8",
            "17 f_user_3d_most_two_view_cate_screening  8",
            "18 f_user_3d_most_three_view_class_screening   8",
            "19 f_user_1w_most_view_item_screening  8",
            "20 f_user_1w_most_view_type_screening  8",
            "21 f_user_1w_most_view_cate_screening  8",
            "22 f_user_1w_most_two_view_cate_screening  8",
            "23 f_user_1w_most_three_view_class_screening   8",
            "24 f_1d_last_view_one_item_screening   8",
            "25 f_1d_last_view_one_type_screening   8",
            "26 f_1d_last_view_two_cates_screening  8",
            "27 f_1d_last_view_three_class_screening    8",
            "28 f_3d_last_view_item_screening   8",
            "29 f_3d_last_view_cates_screening  8",
            "30 f_3d_last_view_two_class_screening  8",
            "31 f_3d_last_view_three_class_screening    8",
            "32 f_1w_last_view_item_screening   8",
            "33 f_1w_last_view_type_screening   8",
            "34 f_1w_last_view_cates_screening  8",
            "35 f_1w_last_view_two_class_screening  8",
            "36 f_all_last_view_two_cates_screening 8",
            "37 f_all_last_view_three_class_screening   8",
            "38 f_user_view_1h_gap_code5_screening  8",
            "39 f_user_view_3d_gap_code5_screening  8",
            "40 f_user_view_1w_gap_code5_screening  8",
            "41 f_user_view_item_trans_code5_screening  8",
            "42 f_user_view_class_trans_code5_screening 8",
            "43 f_user_view_cate_trans_code5_screening  8",
            "44 f_user_view_type_trans_code5_screening  8",
            "45 f_user_2h_most_two_order_type_screening 8",
            "46 f_user_2h_most_order_item_screening 8",
            "47 f_user_12h_most_two_order_cate_screening    8",
            "48 f_user_12h_most_two_cate_screening  8",
            "49 f_user_1w_most_order_item_screening 8",
            "50 f_user_1w_most_thtree_order_cate_screening  8",
            "51 f_user_1w_most_thtree_order_class_screening 8",
            "52 f_user_1m_most_order_item_screening 8",
            "53 f_user_1m_most_order_type_screening 8",
            "54 f_user_1m_most_two_order_cate_screening 8",
            "55 f_user_1m_most_three_order_class_screening  8",
            "56 f_2h_last_order_items_screening 8",
            "57 f_2h_last_order_types_screening 8",
            "58 f_2h_last_order_class_screening 8",
            "59 f_2h_last_order_two_cates_screening 8",
            "60 f_2d_last_order_three_cates_screening   8",
            "61 f_2d_last_order_three_class_screening   8",
            "62 f_2d_last_order_item_screening  8",
            "63 f_12h_last_order_two_types_screening    8",
            "64 f_12h_last_order_item_screening 8",
            "65 f_1w_last_order_items_screening 8",
            "66 f_1w_last_order_cates_screening 8",
            "67 f_1w_last_order_types_screening 8",
            "68 f_1w_last_order_two_cates_screening 8",
            "69 f_1w_last_order_three_class_screening   8",
            "70 f_user_order_1h_gap_code5_screening 8",
            "71 f_user_order_3d_gap_code5_screening 8",
            "72 f_user_order_1w_gap_code5_screening 8",
            "73 f_user_order_item_trans_code5_screening 8",
            "74 f_user_order_class_trans_code5_screening    8",
            "75 f_user_order_cate_trans_code5_screening 8",
            "76 f_user_order_type_trans_code5_screening 8",
            "77 f_ups_age   8",
            "78 f_ups_birthday  8",
            "79 f_ups_birthmonth    8",
            "80 f_ups_birthmonthday 8",
            "81 f_ups_place 8",
            # "82 f_ups_birthprovince 8",
            # "83 f_ups_birthcity 8",
            "84 f_ups_occupation    8",
            "85 f_ups_mobile_type   8",
            "86 f_ups_mobile_os 8",
            "87 f_ups_resident_city 8",
            "88 f_ups_work_geohash  8",
            "89 f_ups_appear_geohash1   8",
            "90 f_ups_appear_geohash2   8",
            "91 f_ups_appear_geohash3   8",
            "92 f_ups_appear_area1  8",
            "93 f_ups_appear_area2  8",
            "94 f_ups_appear_area3  8",
            "95 f_ups_consume_style 8",
            "96 f_ups_saunter_level 8",
            "97 f_ups_pay_cnt   8",
            "98 f_ups_class_preference1 8",
            "99 f_ups_class_preference2 8",
            "100    f_ups_class_preference3 8",
            "101    f_ups_cate_preference1  8",
            "102    f_ups_cate_preference2  8",
            "103    f_ups_cate_preference3  8",
            "104    f_scene_hour    8",
            "105    f_scene_weekday 8",
            "106    f_scene_month   8",
            "107    f_scene_monthday    8",
            "108    f_scene_city    8",
            "109    f_scene_geohash 8",
            "110    f_his_scene_hour0   8",
            "111    f_his_scene_hour1   8",
            "112    f_his_scene_hour2   8",
            "113    f_his_scene_hour3   8",
            "114    f_his_scene_hour4   8",
            "115    f_his_scene_hour5   8",
            "116    f_his_scene_hour6   8",
            "117    f_his_scene_hour7   8",
            "118    f_his_scene_hour8   8",
            "119    f_his_scene_hour9   8",
            "120    f_his_scene_weekday0    8",
            "121    f_his_scene_weekday1    8",
            "122    f_his_scene_weekday2    8",
            "123    f_his_scene_weekday3    8",
            "124    f_his_scene_weekday4    8",
            "125    f_his_scene_weekday5    8",
            "126    f_his_scene_weekday6    8",
            "127    f_his_scene_weekday7    8",
            "128    f_his_scene_weekday8    8",
            "129    f_his_scene_weekday9    8",
            "130    f_his_scene_month0  8",
            "131    f_his_scene_month1  8",
            "132    f_his_scene_month2  8",
            "133    f_his_scene_month3  8",
            "134    f_his_scene_month4  8",
            "135    f_his_scene_month5  8",
            "136    f_his_scene_month6  8",
            "137    f_his_scene_month7  8",
            "138    f_his_scene_month8  8",
            "139    f_his_scene_month9  8",
            "140    f_his_scene_monthday0   8",
            "141    f_his_scene_monthday1   8",
            "142    f_his_scene_monthday2   8",
            "143    f_his_scene_monthday3   8",
            "144    f_his_scene_monthday4   8",
            "145    f_his_scene_monthday5   8",
            "146    f_his_scene_monthday6   8",
            "147    f_his_scene_monthday7   8",
            "148    f_his_scene_monthday8   8",
            "149    f_his_scene_monthday9   8",
            "150    f_viewed_class0 8",
            "151    f_viewed_class1 8",
            "152    f_viewed_class2 8",
            "153    f_viewed_class3 8",
            "154    f_viewed_class4 8",
            "155    f_viewed_cate0  8",
            "156    f_viewed_cate1  8",
            "157    f_viewed_cate2  8",
            "158    f_viewed_cate3  8",
            "159    f_viewed_cate4  8",
            "160    f_viewed_type0  8",
            "161    f_viewed_type1  8",
            "162    f_viewed_type2  8",
            "163    f_viewed_type3  8",
            "164    f_viewed_type4  8",
            "165    f_most_clicked_label    8",
            "166    f_second_clicked_label  8",
            "167    f_1h_clicked_label0 8",
            "168    f_1h_clicked_label1 8",
            "169    f_clicked_label0    8",
            "170    f_clicked_label1    8",
            "171    f_clicked_label2    8",
            # "172    f_last_clicked_label0   8",
            "173    f_last_clicked_label1   8",
            "174    f_last_clicked_label2   8",
            "175    f_has_car   8",
            "176    f_waimai_new_is_pref_geohash    8",
            "177    f_has_child 8",
            "178    f_user_id   8",
            "179    f_user_1w_most_query_type   8",
            "180    f_user_order_brandid_lists_6h   8",
            "181    f_user_order_class_lists_6h 8",
            "182    f_user_order_brandid_lists_24h  8",
            "183    f_all_last_view_cates   8",
            "184    f_user_view_brandid_lists_24h   8",
            "185    f_user_displayed_and_view_cate_id   8",
            "186    f_user_order_cate_lists 8",
            "187    f_user_order_type_lists_6h  8",
            "188    f_user_displayed_and_order_cate_id  8",
            "189    f_user_3d_most_query_three_class    8",
            "190    f_user_view_item_dtype_lists    8",
            "191    f_user_order_type_lists 8",
            "192    f_constellation 8",
            "193    f_user_displayed_and_order_class_id 8",
            "194    f_user_order_class_lists    8",
            "195    f_user_order_item_dtype_lists   8",
            "196    f_user_1d_most_two_query_cate   8",
            "197    f_user_view_type_lists_24h  8",
            "198    f_user_1w_most_query_three_class    8",
            "199    f_user_order_class_lists_24h    8",
            "200    f_user_1w_most_query_two_cate   8",
            "201    f_user_displayed_cate_id    8",
            "202    f_user_order_type_lists_24h 8",
            "203    f_req_page_num  8",
            "204    f_married   8",
            "205    f_user_view_type_lists  8",
            "206    f_user_order_item_dtype_lists_6h    8",
            "207    f_user_1d_most_three_query_class    8",
            "208    f_user_1d_most_query_class  8",
            "209    f_user_displayed_item_num   8",
            "210    f_user_1d_most_query_type   8",
            "211    f_user_order_item_dtype_lists_24h   8",
            "212    f_user_displayed_class_id   8",
            "213    f_user_order_cate_lists_6h  8",
            "214    f_job   8",
            "215    f_user_displayed_and_view_class_id  8",
            "216    f_user_view_brandid_lists_6h    8",
            "217    f_user_view_item_dtype_lists_6h 8",
            "218    f_user_view_cate_lists  8",
            "219    f_user_view_class_lists_24h 8",
            "220    f_user_dislayed_and_view_item_num   8",
            "221    f_user_order_cate_lists_24h 8",
            "222    f_user_3d_most_query_two_cate   8",
            "223    f_user_displayed_and_order_item_num 8",
            "224    f_gender    8",
            "225    f_age   8",
            "226    f_user_view_cate_lists_24h  8",
            "227    f_user_view_class_lists_6h  8",
            "228    f_user_view_cate_lists_6h   8",
            "229    f_user_order_brandid_lists  8",
            "230    f_user_3d_most_query_type   8",
            "231    f_user_view_item_dtype_lists_24h    8",
            "232    f_ubid_1day 8",
            "233    f_ubid_cnt_1day 8",
            "234    f_ubid_7days    8",
            "235    f_ubid_cnt_7days    8",
            "236    f_ubid_14days   8",
            "237    f_ubid_cnt_14days   8",
            "238    f_ubid_30days   8",
            "239    f_ubid_cnt_30days   8",
            "240    f_ubid_180days  8",
            "241    f_ubid_cnt_180days  8",
            # "242    f_app_source    8",
            "243    f_weather   8",
            "244    f_cityid    8",
            "245    f_geohash   8",
            "246    f_app_version   8",
            "247    f_loc_cityid    8",
            "248    f_agent 8",
            "249    f_client_type   8",
            "250    f_remote_type   8",
            "251    f_reslution 8"
        ],
        "new_user_sparse_list": [
            # "600 f_user_id_log   8",
            # "608    f_distance  8",
            # "609    f_adistance 8",
            # "610    f_cdistance 8",
            "611    f_scene_daytype 8",
            "612    f_scene_daynames    8",
            "613    f_weather_iconid    8",
            "614    f_weather_temp  8",
            "615    f_weather_temphigh  8",
            "616    f_weather_templow   8",
            "617    f_weather_pm25  8",
            "618    f_weather_zswd  8",
            "619    f_weather_windlevel 8",
            "620    f_weather_temprange 8",
            "621    f_sceneworkdaycityhour_mtorder7dtop5cates   8",
            "622    f_sceneworkdaycityhour_mtorder30dtop5cates  8",
            "623    f_sceneweathercity_mtorder30dtop5cates  8",
            "624    f_scenetemperaturecity_mtorder30dtop5cates  8"
        ],
        "item_sparse_list": [
            "252    f_item_type_id  8",
            "253    f_item_avg_price    8",
            "254    f_class 8",
            "255    f_avg_delivery_time 8",
            # "256    f_dtype 8",
            "257    f_type  8",
            "258    f_price 8",
            # "259    f_shipping_fee  8",
            "260    f_discount  8",
            "261    f_ratecount 8",
            "262    f_brandid   8",
            "263    f_poi_lowestpirce   8",
            "264    f_cate  8",
            "265    f_avg_amount    8",
            # "266    f_gcomment_ratio    8",
            "267    f_avgpriceperperson 8",
            "268    f_exp_item_30days_exposure_cnt_1    8",
            "269    f_poi_discount  8",
            # "270    f_meal_count    8",
            "271    f_online_days   8"
        ],
        "new_item_sparse_list": [
            "272    f_item_new_cate_name    8",
            "273    f_item_geohash5s    8",
            "274    f_item_geohash6s    8",
            # "601    f_item_basic_type 8",
            "602    f_item_dtype    8",
            "603    f_bu_id 8",
            "604    f_first_cate    8",
            "605    f_second_cate   8",
            "606    f_third_cate    8",
            "607    f_label_bucket_price    8"
        ],
        "user_d2s_sparse_list": [
            "700    f_ud2s_0_trip_geohash_ord   8",
            "701    f_ud2s_1_csscore_top3   8",
            "702    f_ud2s_2_csscore_top10  8",
            "703    f_ud2s_3_csscore_top30  8",
            # "704    f_ud2s_4_hp_one_days_click_cnt  8",
            # "705    f_ud2s_5_hp_cur_day_is_clicked  8",
            "706    f_ud2s_6_hotel_intention_wilson 8",
            "707    f_ud2s_7_hotel_geohash_clk_coec 8",
            # "708    f_ud2s_8_hp_seven_days_is_clicked   8",
            "709    f_ud2s_9_waimai_geohash_clk 8",
            "710    f_ud2s_10_hotel_geohash_exp 8",
            "711    f_ud2s_11_avg_total_click_num   8",
            "712    f_ud2s_12_hotel_geohash_clk 8",
            "713    f_ud2s_13_waimai_geohash_exp    8",
            "714    f_ud2s_14_other_geohash_clk 8",
            "715    f_ud2s_15_avg_total_page_num    8",
            "716    f_ud2s_16_avg_page_num_before_order 8",
            "717    f_ud2s_17_other_geohash_exp 8",
            "718    f_ud2s_18_order_day_times_mt    8",
            "719    f_ud2s_19_waimai_new_pref_geohash_ratio 8",
            "720    f_ud2s_20_trip_geohash_exp  8",
            "721    f_ud2s_21_meishi_geohash_clk_coec   8",
            "722    f_ud2s_22_avg_pay_amount    8",
            "723    f_ud2s_23_trip_geohash_clk  8",
            "724    f_ud2s_24_waimai_geohash_ord    8",
            "725    f_ud2s_25_avg_first_cate_num_before_order   8",
            "726    f_ud2s_26_avg_second_cate_num_before_order  8",
            "727    f_ud2s_27_other_geohash_clk_coec    8",
            "728    f_ud2s_28_meishi_geohash_clk    8",
            "729    f_ud2s_29_meishi_geohash_exp    8",
            "730    f_ud2s_30_waimai_geohash_clk_coec   8",
            "731    f_ud2s_31_avg_firstclick_order_time 8",
            "732    f_ud2s_32_waimai_geohash_ord_coec   8",
            # "733    f_ud2s_33_hp_seven_days_click_cnt   8",
            "734    f_ud2s_34_trip_geohash_clk_coec 8",
            "735    f_ud2s_35_avg_click_num_after_order 8",
            "736    f_ud2s_36_trip_intention_wilson 8",
            # "737    f_ud2s_37_hp_cur_day_click_cnt  8",
            "738    f_ud2s_38_hotel_geohash_ord_coec    8",
            "739    f_ud2s_39_order_day_times_guesslike 8",
            "740    f_ud2s_40_meishi_geohash_ord_coec   8",
            "741    f_ud2s_41_meishi_geohash_ord    8",
            # "742    f_ud2s_42_area_amount   8",
            "743    f_ud2s_43_numresults    8",
            # "744    f_ud2s_44_hp_clicked_passed_time    8",
            "745    f_ud2s_45_trip_intention    8",
            "746    f_ud2s_46_other_geohash_ord_coec    8",
            "747    f_ud2s_47_meishi_intention_wilson   8",
            "748    f_ud2s_48_meishi_intention  8",
            "749    f_ud2s_49_avg_page_num_after_order  8",
            "750    f_ud2s_50_poi_intention 8",
            "751    f_ud2s_51_other_geohash_ord 8",
            # "752    f_ud2s_52_hp_one_days_is_clicked    8",
            "753    f_ud2s_53_hotel_intention   8",
            "754    f_ud2s_54_deal_intention    8",
            "755    f_ud2s_55_avg_firstpage_order_time  8",
            "756    f_ud2s_56_avg_click_num_before_order    8",
            "757    f_ud2s_57_poi_intention_wilson  8",
            "758    f_ud2s_58_deal_intention_wilson 8",
            "759    f_ud2s_59_user_level    8",
            "760    f_ud2s_60_user_feedback_score   8",
            "761    f_ud2s_61_isremote  8",
            "762    f_ud2s_62_client_android    8",
            "763    f_ud2s_63_client_iphone 8",
            "764    f_ud2s_64_userid0   8",
            "765    f_ud2s_65_user_rtgeo_pref   8",
            "766    f_ud2s_66_user_home_dis 8",
            "767    f_ud2s_67_user_work_dis 8",
            "768    f_ud2s_68_user_consume_dis  8",
            "769    f_ud2s_69_user_all_view_num 8",
            "770    f_ud2s_70_user_all_order_num    8",
            "771    f_ud2s_71_user_view_stay_time_sum   8",
            "772    f_ud2s_72_user_view_stay_time_avg   8",
            "773    f_ud2s_73_user_exposure_avg_pos 8",
            "774    f_ud2s_74_user_click_avg_pos    8",
            "775    f_ud2s_75_user_order_avg_pos    8",
            "776    f_ud2s_76_user_pay_avg_pos  8",
            "777    f_ud2s_77_user_all_exposure_cnt 8",
            "778    f_ud2s_78_user_view_depth   8",
            "779    f_ud2s_79_exp_user_30days_exposure_cnt  8",
            "780    f_ud2s_80_exp_user_30days_ctr   8",
            "781    f_ud2s_81_exp_user_30days_cvr   8",
            "782    f_ud2s_82_exp_user_30days_click_cnt 8",
            "783    f_ud2s_83_exp_user_30days_cxr   8",
            "784    f_ud2s_84_exp_user_30days_order_cnt 8",
            "785    f_ud2s_85_consume_user_orders   8",
            "786    f_ud2s_86_consume_user_coupons  8",
            "787    f_ud2s_87_consume_user_coupons_avg  8",
            "788    f_ud2s_88_consume_user_moneys   8",
            "789    f_ud2s_89_consume_user_moneys_avg   8",
            "790    f_ud2s_90_consume_user_avg_time 8",
            "791    f_ud2s_91_consume_user_middle_time  8",
            "792    f_ud2s_92_consume_user_consume_ords 8",
            "793    f_ud2s_93_consume_user_noconsume_ords   8",
            "794    f_ud2s_94_consume_user_consume_ratio    8",
            "795    f_ud2s_95_user_30days_time_decay_click_wcnt 8",
            "796    f_ud2s_96_user_30days_time_decay_order_wcnt 8",
            "797    f_ud2s_97_user_30days_time_decay_impression_wcnt    8",
            "798    f_ud2s_98_user_30days_time_decay_wctr   8",
            "799    f_ud2s_99_user_30days_time_decay_wcvr   8",
            "800    f_ud2s_100_user_30days_time_decay_wcxr  8",
            "801    f_ud2s_101_ups_gender   8",
            "802    f_ud2s_102_ups_edu  8",
            "803    f_ud2s_103_ups_salary   8",
            "804    f_ud2s_104_ups_married  8",
            "805    f_ud2s_105_ups_child    8",
            "806    f_ud2s_106_ups_promotion    8",
            "807    f_ud2s_107_ups_activity 8",
            "808    f_ud2s_108_clicked_cnt_label0   8",
            "809    f_ud2s_109_clicked_cnt_label1   8",
            "810    f_ud2s_110_clicked_cnt_label2   8",
            "811    f_ud2s_111_clicked_cnt_label3   8",
            # "812    f_ud2s_112_clicked_cnt_label4   8",
            "813    f_ud2s_113_clicked_cnt_label5   8",
            "814    f_ud2s_114_clicked_cnt_label6   8",
            "815    f_ud2s_115_clicked_cnt_label7   8",
            "816    f_ud2s_116_clicked_cnt_label8   8",
            "817    f_ud2s_117_clicked_cnt_label9   8",
            "818    f_ud2s_118_clicked_label0_passed_time   8",
            "819    f_ud2s_119_clicked_label1_passed_time   8",
            "820    f_ud2s_120_clicked_label2_passed_time   8",
            "821    f_ud2s_121_clicked_label3_passed_time   8",
            # "822    f_ud2s_122_clicked_label4_passed_time   8",
            "823    f_ud2s_123_clicked_label5_passed_time   8",
            "824    f_ud2s_124_clicked_label6_passed_time   8",
            "825    f_ud2s_125_clicked_label7_passed_time   8",
            "826    f_ud2s_126_clicked_label8_passed_time   8",
            "827    f_ud2s_127_clicked_label9_passed_time   8"
            # "828    f_ud2s_128_user_item_mt_view_cnt    8",
            # "829    f_ud2s_129_user_item_mt_order_cnt   8",
            # "830    f_ud2s_130_s_viewed_item_passed_time    8",
            # "831    f_ud2s_131_s_viewed_dtype_passed_time   8",
            # "832    f_ud2s_132_s_viewed_class_passed_time   8",
            # "833    f_ud2s_133_s_viewed_cate_passed_time    8",
            # "834    f_ud2s_134_s_viewed_type_passed_time    8",
            # "835    f_ud2s_135_s_ordered_item_passed_time   8",
            # "836    f_ud2s_136_s_ordered_dtype_passed_time  8",
            # "837    f_ud2s_137_s_ordered_class_passed_time  8",
            # "838    f_ud2s_138_s_ordered_cate_passed_time   8",
            # "839    f_ud2s_139_s_ordered_type_passed_time   8"
        ],
        "user_item_d2s_sparse_list": [
            "828    f_ud2s_128_user_item_mt_view_cnt    8",
            "829    f_ud2s_129_user_item_mt_order_cnt   8",
            "830    f_ud2s_130_s_viewed_item_passed_time    8",
            "831    f_ud2s_131_s_viewed_dtype_passed_time   8",
            "832    f_ud2s_132_s_viewed_class_passed_time   8",
            "833    f_ud2s_133_s_viewed_cate_passed_time    8",
            "834    f_ud2s_134_s_viewed_type_passed_time    8",
            "835    f_ud2s_135_s_ordered_item_passed_time   8",
            "836    f_ud2s_136_s_ordered_dtype_passed_time  8",
            "837    f_ud2s_137_s_ordered_class_passed_time  8",
            "838    f_ud2s_138_s_ordered_cate_passed_time   8",
            "839    f_ud2s_139_s_ordered_type_passed_time   8"
        ],
        "item_d2s_sparse_list": [
            "840    f_id2s_0_avgpriceperperson  8",
            "841    f_id2s_1_avg_amount 8",
            # "842    f_id2s_2_avg_amount_month   8",
            # "843    f_id2s_3_avg_delivery_time  8",
            # "844    f_id2s_4_avg_fact_amount    8",
            # "845    f_id2s_5_avg_fact_amount_month  8",
            # "846    f_id2s_6_brand_click_coec   8",
            "847    f_id2s_7_brand_grade    8",
            # "848    f_id2s_8_brand_order_coec   8",
            "849    f_id2s_9_class_click_coec   8",
            "850    f_id2s_10_click_num_city    8",
            # "851    f_id2s_11_click_num_time_10_11  8",
            "852    f_id2s_12_comment_num   8",
            "853    f_id2s_13_consume_item_consume_ords 8",
            "854    f_id2s_14_consume_item_consume_ratio    8",
            "855    f_id2s_15_consume_item_coupons_avg  8",
            "856    f_id2s_16_consume_item_middle_time  8",
            "857    f_id2s_17_consume_item_moneys_avg   8",
            "858    f_id2s_18_consume_item_noconsume_ords   8",
            "859    f_id2s_19_deal_poi_num  8",
            "860    f_id2s_20_delivery_avg_score    8",
            # "861    f_id2s_21_discount  8",
            "862    f_id2s_22_exp_item_30days_click_cnt_1   8",
            "863    f_id2s_23_exp_item_30days_ctr_1 8",
            "864    f_id2s_24_exp_item_30days_cxr_1 8",
            "865    f_id2s_25_exp_item_30days_exposure_cnt_1    8",
            "866    f_id2s_26_exp_item_30days_order_cnt_1   8",
            "867    f_id2s_27_exp_item_top10_ctr_1  8",
            "868    f_id2s_28_exp_item_top10_exposure_cnt_1 8",
            "869    f_id2s_29_exp_item_top1_click_cnt_1 8",
            "870    f_id2s_30_exp_item_top1_ctr_1   8",
            "871    f_id2s_31_exp_item_top5_click_cnt_1 8",
            "872    f_id2s_32_food_avg_score    8",
            # "873    f_id2s_33_gcomment_ratio    8",
            # "874    f_id2s_34_guess_lastclick_item_first_clicks 8",
            "875    f_id2s_35_guess_lastclick_item_first_ctr_smooth 8",
            "876    f_id2s_36_guess_lastclick_item_last_clicks  8",
            "877    f_id2s_37_guess_lastclick_item_last_ctr_smooth  8",
            "878    f_id2s_38_guess_lastclick_item_last_ratio   8",
            "879    f_id2s_39_guess_lastclick_item_last_ratio_smooth    8",
            "880    f_id2s_40_hp_one_day_click_cnt_1    8",
            # "881    f_id2s_41_hp_third_day_click_cnt_1  8",
            # "882    f_id2s_42_image_feature_27  8",
            # "883    f_id2s_43_image_feature_28  8",
            # "884    f_id2s_44_image_feature_32  8",
            # "885    f_id2s_45_image_feature_40  8",
            # "886    f_id2s_46_image_feature_42  8",
            # "887    f_id2s_47_image_feature_48  8",
            # "888    f_id2s_48_image_feature_7   8",
            # "889    f_id2s_49_image_feature_9   8",
            # "890    f_id2s_50_ismulticity   8",
            "891    f_id2s_51_item_30days_time_decay_click_wcnt 8",
            "892    f_id2s_52_item_30days_time_decay_impression_wcnt    8",
            "893    f_id2s_53_item_30days_time_decay_wctr   8",
            "894    f_id2s_54_item_30days_time_decay_wcvr   8",
            "895    f_id2s_55_item_avg_price    8",
            "896    f_id2s_56_item_click_avg_pos    8",
            "897    f_id2s_57_item_click_coec   8",
            "898    f_id2s_58_item_exposure_avg_pos 8",
            "899    f_id2s_59_item_order_coec   8",
            "900    f_id2s_60_item_pay_avg_pos  8",
            "901    f_id2s_61_item_show_avg_pos 8",
            "902    f_id2s_62_meal_count    8",
            "903    f_id2s_63_min_price 8",
            # "904    f_id2s_64_mt_one_day_click_cnt_1    8",
            "905    f_id2s_65_mt_third_day_click_cnt_1  8",
            "906    f_id2s_66_new_ctr   8",
            "907    f_id2s_67_new_cvr   8",
            "908    f_id2s_68_online_days   8",
            "909    f_id2s_69_ordercount    8",
            "910    f_id2s_70_poi_discount  8",
            "911    f_id2s_71_poi_historycouponcount    8",
            "912    f_id2s_72_poi_latestweekcoupon  8",
            "913    f_id2s_73_poi_lowestpirce   8",
            "914    f_id2s_74_poi_marknumbers   8",
            "915    f_id2s_75_poi_open_days 8",
            "916    f_id2s_76_poi_score 8",
            "917    f_id2s_77_poi_scoreratio    8",
            "918    f_id2s_78_price 8",
            "919    f_id2s_79_ratecount 8",
            "920    f_id2s_80_rec_num_time_10_11    8",
            "921    f_id2s_81_rec_num_time_20_9 8",
            "922    f_id2s_82_search_lastclick_item_clicks  8",
            "923    f_id2s_83_search_lastclick_item_exps    8",
            "924    f_id2s_84_search_lastclick_item_first_ratio_smooth  8",
            "925    f_id2s_85_search_lastclick_item_last_clicks 8",
            "926    f_id2s_86_search_lastclick_item_last_ctr    8",
            "927    f_id2s_87_search_lastclick_item_last_ratio_smooth   8",
            "928    f_id2s_88_shipping_fee  8",
            # "929    f_id2s_89_shipping_meituan  8",
            # "930    f_id2s_90_waimai_ctr    8",
            # "931    f_id2s_91_waimai_cvr    8",
            # "932    f_id2s_92_waimai_cxr    8",
            # "933    f_id2s_93_waimai_new_avg_amount_month   8",
            "934    f_id2s_94_waimai_new_avg_delivery_time  8",
            "935    f_id2s_95_waimai_new_avg_manjian_discount   8",
            "936    f_id2s_96_waimai_new_comment_num    8",
            "937    f_id2s_97_waimai_new_ctr    8",
            "938    f_id2s_98_waimai_new_cvr    8",
            "939    f_id2s_99_waimai_new_cxr    8",
            "940    f_id2s_100_waimai_new_discount  8",
            "941    f_id2s_101_waimai_new_fact_avg_amount_month 8",
            "942    f_id2s_102_waimai_new_good_comment_ratio    8",
            "943    f_id2s_103_waimai_new_ol_time_7days 8",
            "944    f_id2s_104_waimai_new_order_num_day 8",
            "945    f_id2s_105_waimai_new_order_num_month   8",
            "946    f_id2s_106_waimai_new_poi_open_days 8"
        ],
        "dianshang_item_sparse_list": [
            "947    f_sku_mall_cat1_pay_num_sparse_30   8",
            "948    f_sku_mall_spu_pay_num_sparse_30    8",
            "949    f_sku_mall_cat3_clk_num_sparse_7    8",
            "950    f_sku_mall_cat2_order_num_sparse_7  8",
            "951    f_shangou_origin_brand_id   8",
            "952    f_sku_mall_cat3_order_num_sparse_30 8",
            "953    f_em_third_cate_id  8",
            "954    f_sku_mall_cat3_clk_num_sparse_30   8",
            "955    f_shangou_upc_code  8",
            "956    f_youxuan_commonid_7days_exp_log2   8",
            "957    f_youxuan_3rdcate_3days_pay_log2    8",
            "958    f_sku_mall_cat1_clk_num_sparse_7    8",
            "959    f_sku_mall_cat2_pay_num_sparse_30   8",
            "960    f_youxuan_commonid_7days_pay_log2   8",
            "961    f_sku_mall_cat1_pay_num_sparse_7    8",
            "962    f_sku_mall_cat3_exp_num_sparse_7    8",
            "963    f_sku_mall_cat4_clk_num_sparse_30   8",
            "964    f_sku_mall_cat1_order_num_sparse_7  8",
            "965    f_sku_mall_cat1_exp_num_sparse_30   8",
            "966    f_youxuan_3rdcate_3days_exp_log2    8",
            "967    f_sku_mall_spu_pay_num_sparse_7 8",
            "968    f_sku_mall_cat3_pay_num_sparse_30   8",
            "969    f_sku_mall_cat4_pay_num_sparse_7    8",
            "970    f_sku_mall_cat2_exp_num_sparse_30   8",
            "971    f_sku_mall_spu_clk_num_sparse_7 8",
            "972    f_sku_mall_cat2_order_num_sparse_30 8",
            "973    f_youxuan_3rdcate_3days_click_log2  8",
            "974    f_sku_mall_cat4_clk_num_sparse_7    8",
            "975    f_youxuan_commonid_3days_pay_log2   8",
            "976    f_sku_mall_spu_exp_num_sparse_30    8",
            "977    f_yiyao_brand_id    8",
            "978    f_youxuan_3rdcate_7days_click_log2  8",
            "979    f_sku_mall_cat1_exp_num_sparse_7    8",
            "980    f_yiyao_poi_id  8",
            "981    f_sku_mall_cat1_order_num_sparse_30 8",
            "982    f_sku_mall_spu_exp_num_sparse_7 8",
            "983    f_youxuan_commonid_3days_exp_log2   8",
            "984    f_sku_mall_cat2_pay_num_sparse_7    8",
            "985    f_sku_mall_cat4_exp_num_sparse_7    8",
            "986    f_sku_mall_cat1_clk_num_sparse_30   8",
            "987    f_sku_mall_cat4_pay_num_sparse_30   8",
            "988    f_youxuan_common_feature_id 8",
            "989    f_em_item_sale_price_sparse 8",
            "990    f_youxuan_commonid_3days_click_log2 8",
            "991    f_sku_mall_cat2_clk_num_sparse_7    8",
            "992    f_sku_mall_cat3_exp_num_sparse_30   8",
            "993    f_sku_mall_cat4_order_num_sparse_7  8",
            "994    f_youxuan_commonid_7days_click_log2 8",
            "995    f_sku_mall_cat2_clk_num_sparse_30   8",
            "996    f_sku_mall_spu_order_num_sparse_30  8",
            "997    f_sku_mall_spu_order_num_sparse_7   8",
            "998    f_youxuan_3rdcate_7days_exp_log2    8",
            "999    f_em_first_cate_id  8",
            "1000   f_em_forth_cate_id  8",
            "1001   f_sku_mall_spu_clk_num_sparse_30    8",
            "1002   f_em_second_cate_id 8",
            "1003   f_sku_mall_cat2_exp_num_sparse_7    8",
            "1004   f_sku_mall_cat4_exp_num_sparse_30   8",
            "1005   f_youxuan_3rdcate_7days_pay_log2    8",
            "1006   f_sku_mall_cat3_order_num_sparse_7  8",
            "1007   f_yiyao_upc_code    8",
            "1008   f_sku_mall_cat3_pay_num_sparse_7    8",
            "1009   f_item_id   8",
            "1010   f_sku_mall_cat4_order_num_sparse_30 8",
            "1011   f_shangou_poi_id    8",
            "1012   f_shangou_sold_product_num_30d_log2 8",
            "1013   f_yiyao_day30_sale_cnt_log2 8"
        ]
    }
}

##双塔fc
global g_mlx_embed_names
global g_mlx_embed_mapping

global g_my_name_collections
global g_my_name_mapping
global extra_graph_states
##定义路径状态，目前predict状态和 predict stage状态冲突，predict状态可以使用自定义的状态如('PREDICT','default')替代
extra_graph_states = [
    graph_state_helper('TRAIN'),
    graph_state_helper('EVALUATE'),
    graph_state_helper('PREDICT', 'default'),
    graph_state_helper('PREDICT', 'item_precompute'),
    graph_state_helper('PREDICT', 'user_precompute'),
    graph_state_helper('PREDICT', 'user_item_precompute'),
    graph_state_helper('PREDICT', 'evaluate_ctr_cvr'),
    graph_state_helper('PREDICT', 'predict_top')
]

##定义初始状态参数，dense中user和item，context的个数，以及user塔和item塔的隐藏层的维度

# user_dense_len = int(conf['dense']['user_dense_len']) #128
# item_dense_len = int(conf['dense']['item_dense_len']) #107
# user_item_dense_len = int(conf['dense']['user_item_dense_len']) #12
user_dense_len = 128
item_dense_len = 107
user_item_dense_len = 12
user_hidden_layer_dim = [300]
item_hidden_layer_dim = [300]
user_item_hidden_layer_dim = [300]

student_weight_factor = 0.9


def build_model():
    # handle dense

    hidden_num = 300
    offline_only_dense_len = user_dense_len + item_dense_len + user_item_dense_len * 2
    online_only_dense_len = 3 * hidden_num

    dense_input = tfmlx.get_dense(cid=1024, dense_len=max(offline_only_dense_len, online_only_dense_len))

    # handle sparse
    user_sparse_list = conf['sparse']['user_sparse_list']
    new_user_sparse_list = conf['sparse']['new_user_sparse_list']
    item_sparse_list = conf['sparse']['item_sparse_list']
    new_item_sparse_list = conf['sparse']['new_item_sparse_list']
    user_d2s_sparse_list = conf['sparse']['user_d2s_sparse_list']
    item_d2s_sparse_list = conf['sparse']['item_d2s_sparse_list']
    dianshang_item_sparse_list = conf['sparse']['dianshang_item_sparse_list']

    # 定义sparse特征
    SPARSE_FEATURES = user_sparse_list + new_user_sparse_list + item_sparse_list + new_item_sparse_list+dianshang_item_sparse_list
    sparse_features = []
    for fea_info in SPARSE_FEATURES:
        cid = int(fea_info.split()[0])
        name = str(fea_info.split()[1])
        feature = tfmlx.create_sparse_feature(cid, name)
        sparse_features.append(feature)
    # 为sparse特征添加linear部分
    # 目前由于框架限制，sparse特征只有添加了linear部分才可以创建embedding
    # 即tfmlx.linear_column()等价于原生MLX版本中的m.add_col()
    linear_column = tfmlx.linear_column(sparse_features)

    global g_mlx_embed_mapping
    global g_mlx_embed_names
    global g_my_name_collections

    g_mlx_embed_mapping = {}
    g_mlx_embed_names = []
    g_my_name_collections = {}
    g_my_name_collections["user_emb"] = []
    # g_my_name_collections["context_emb"] = []
    g_my_name_collections["item_emb"] = []

    # 这是一个简单的API封装，行为和原生MLX版本的m.add_col_embed类似，便于原生MLX模型代码的重用
    def add_embed(name, embed_len, flag):
        f = tfmlx.get_sparse_feature_by_name(name)
        # 对稀疏特征创建Embedding定义，仅定义Embedding维度即可
        emb_var = tfmlx.embedding_column(
            f, dimensions=embed_len, combiner='sum', use_cvm=True)
        # 使用get_output方法获得输入特征经Embedding转换后且经过combiner操作后的实际数值
        # 注意：与TF需要自己处理输入数据并进行embedding_lookup不同，这里不涉及任何对原始输入数据的操作
        # 原始输入数据（如libsvm格式数据）的解析和embedding填充是在MLX框架内部完成的
        emb_output = emb_var.get_output()
        # 加入全局的映射表，后面会在TF模型中用到
        g_mlx_embed_mapping[emb_output.name] = emb_output
        # 加入全局列表，用于传入TF Model Wrapper
        g_mlx_embed_names.append(emb_output.name)
        g_my_name_collections[flag].append(emb_output.name)
        # return emb_output

    # user emb
    for fea_info in user_sparse_list + new_user_sparse_list:
        name = str(fea_info.split()[1])
        dim_len = int(fea_info.split()[2])
        add_embed(name, dim_len, "user_emb")
    # item emb
    for fea_info in item_sparse_list + new_item_sparse_list + dianshang_item_sparse_list:
        name = str(fea_info.split()[1])
        dim_len = int(fea_info.split()[2])
        add_embed(name, dim_len, "item_emb")

    label0 = tfmlx.get_labels(indexes=[0], name="label0")
    label1 = tfmlx.get_labels(indexes=[1], name="label1")
    labels = [label0, label1]
    sample_weight0 = tfmlx.get_sample_weights(indexes=[0], name="sample_weight0")
    sample_weight1 = tfmlx.get_sample_weights(indexes=[1], name="sample_weight1")
    sample_weights = [sample_weight0, sample_weight1]

    is_training = True
    # 调用TensorFlow创建模型计算图
    # 所有TensorFlow的API只能在这个函数里使用！
    # 这个函数外面的tfmlx API都是MLX的API，不是TensorFlow的API！！！
    model(
        dense_input,
        # 注意1：这里输入用户定义的Embedding时，必须保证顺序!！
        # 注意2：这里传入的变量不能重复！！不能重复！！
        [g_mlx_embed_mapping[x] for x in g_mlx_embed_names],
        labels,
        sample_weights,
        is_training
    )

    # 创建优化器,此处定义的参数仍然可以被外部训练配置覆盖
    adam_opt = tfmlx.Adam(1e-5, l2_regularization=1e-7)
    ftrl_opt = tfmlx.Ftrl(1e-4, l1_regularization=1e-6)
    cvm_sadam_opt = tfmlx.CvmSadam(learning_rate=1e-5, l2_regularization=1e-7)

    ftrl_opt.optimize(tfmlx.LINEAR_VARIABLES)
    adam_opt.optimize(tfmlx.GRAPH_VARIABLES)
    cvm_sadam_opt.optimize(tfmlx.EMBEDDING_VARIABLES)
    tfmlx.set_filter(capacity=(1 << 23), min_cnt=5)
    tfmlx.set_col_max_train_epoch(1)

    return tfmlx.get_model()


# user graph
# item graph
# ui graph
# element sum
# fc -> 1
# ctr loss
# cvr loss
@tfmlx.tf_wrapper(extra_graph_states, no_default_states=True)
def model(dense_input, embeddings, labels, sample_weights, is_training, **kwargs):
    # dense_input, linear, embeddings, ...:
    #   所有使用MLX API得到的变量都必须作为参数传入该函数才能使用
    # is_training: 用于占位的参数，用户不需要指定参数值，可以理解为一个特殊的"placeholder"
    #   此参数用于训练/预测时计算方式不同的算子，例如：
    #   tf.layers.dropout, tf.layers.batch_normalization
    #   这类算子都有指示是否处于训练状态的标志位training，用户直接将该占位符填入即可
    #   MLX API在生成计算图时，会根据训练/预测状态给该占位参数填入正确的值

    # 此处完成原生MLX模型m.add_input的功能
    def add_input_layer(input_src, scope_name='default_input'):
        with tf.variable_scope(scope_name):
            input_dim = int(input_src.shape[1])
            weight_1 = tf.get_variable(
                "input_w_1", [1, input_dim],
                # 此处使用MLX xavier_initializer，mode设为COUNT_COL则行为与原生MLX保持一致
                initializer=tfmlx.xavier_initializer(mode='COUNT_COL')
            )
            weight_2 = tf.get_variable(
                "input_w_2", [1, input_dim],
                initializer=tfmlx.xavier_initializer(mode='COUNT_COL')
            )
            zeros = tf.cast(tf.less(input_src, 1e-6), tf.float32)
            output = tf.multiply(input_src, weight_1) + \
                     tf.multiply(zeros, weight_2)
            return output

    def add_sum(name, inputs):
        return tf.add_n(inputs, name=name)

    def add_fc(input_var, units, activation=None, name='fc'):
        with tf.variable_scope('foo', reuse=tf.AUTO_REUSE):
            input_dim = int(input_var.shape[1])
            var_w = tf.get_variable('%s_w' % name, [units, input_dim],
                                    initializer=tfmlx.xavier_initializer(mode='COUNT_COL'))
            var_b = tf.get_variable('%s_b' % name, [1, units], initializer=tf.zeros_initializer())
            h = tf.matmul(input_var, tf.transpose(var_w)) + var_b
            if activation:
                return activation(h)
        return h

    global g_mlx_embed_names
    # 这里建立(MLX变量名->TF变量)和(TF变量名->MLX变量名)的映射关系,便于使用
    mlx2tf_embed_mapping = {
        x[0]: x[1] for x in zip(g_mlx_embed_names, embeddings)
    }
    tf2mlx_embed_name_mapping = {  # noqa
        x[1].name: x[0] for x in zip(g_mlx_embed_names, embeddings)
    }

    def get_my_collection(name):
        global g_my_name_collections
        if name not in g_my_name_collections:
            raise Exception('Cannot find collection: ' + name)
        return [mlx2tf_embed_mapping[x] for x in g_my_name_collections[name]]

    label0, label1 = labels[0], labels[1]
    sample_weight0, sample_weight1 = sample_weights[0], sample_weights[1]

    dense_output = add_input_layer(dense_input)

    user_dense_gather = [0, 1, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                         28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56,
                         57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                         104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124,
                         125, 126, 127]
    item_dense_gather = [0, 1, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                         35, 36, 37, 38, 39, 40, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69,
                         70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 94, 95, 96, 97, 98,
                         99, 100, 101, 102, 103, 104, 105, 106]

    ##user-part
    user_dense_input = tf.slice(dense_output, [0, 0], [-1, user_dense_len])
    # print('user_dense_input shape: ', user_dense_input.shape)
    # user_dense_input = tf.slice(dense_output, [0], [user_dense_len])
    user_dense_gather_input = tf.gather(user_dense_input, axis=1, indices=user_dense_gather)
    user_embeds = get_my_collection('user_emb')
    # print('user_embeds: ',user_embeds)
    # user_embeds_print = tf.Print(user_embeds, [user_dense_gather_input, user_embeds], message='user input:', summarize=1000)
    user_input = tf.concat([user_dense_gather_input] + user_embeds, axis=1)
    print('user_input embeds: ', user_input)
    for i, units in enumerate(user_hidden_layer_dim):
        user_name = 'user_fc%d' % units
        act = tf.nn.relu
        # if i < len(user_hidden_layer_dim) - 1:
        #     act = tf.nn.relu
        user_input = add_fc(user_input, units, act, user_name)

    ##ui-part
    user_item_dense_input = tf.slice(dense_output, [0, user_dense_len], [-1, user_item_dense_len])
    user_item_mask = tf.slice(dense_input, [0, user_dense_len + user_item_dense_len], [-1, user_item_dense_len])
    user_item_input = tf.multiply(user_item_dense_input, user_item_mask)
    # user_item_dense_input = tf.slice(dense_output, [item_dense_len], [user_item_dense_len])

    for i, units in enumerate(user_item_hidden_layer_dim):
        user_item_name = 'user_item_fc%d' % units
        act = None
        if i < len(user_item_hidden_layer_dim) - 1:
            act = tf.nn.relu
        user_item_input = add_fc(user_item_input, units, act, user_item_name)

    ##item-part
    item_dense_input = tf.slice(dense_output, [0, user_dense_len + 2 * user_item_dense_len], [-1, item_dense_len])
    # item_dense_input = tf.slice(dense_output, [user_dense_len], [item_dense_len])
    item_dense_gather_input = tf.gather(item_dense_input, axis=1, indices=item_dense_gather)
    item_embeds = get_my_collection('item_emb')
    # item_embeds_print = tf.Print(item_embeds, [item_dense_gather_input, item_embeds], message='item input:', summarize=1000)
    # print('item embeds: ',item_embeds)
    item_input = tf.concat([item_dense_gather_input] + item_embeds, axis=1)
    for i, units in enumerate(item_hidden_layer_dim):
        item_name = 'item_fc%d' % units
        act = tf.nn.relu
        # if i < len(item_hidden_layer_dim) - 1:
        #     act = tf.nn.relu
        item_input = add_fc(item_input, units, act, item_name)

    ##top-part
    ##predict内部使用先定义，后使用
    user_top_input = tf.slice(user_input, [0, 0], [-1, user_hidden_layer_dim[-1]])
    item_top_input = tf.slice(item_input, [0, 0], [-1, item_hidden_layer_dim[-1]])
    user_item_top_input = tf.slice(user_item_input, [0, 0], [-1, user_item_hidden_layer_dim[-1]])

    top_fc_input = tf.concat([user_top_input, item_top_input, user_item_top_input], axis=1)
    # top_input_pre = DNN(hidden_units=[300], name='dnn_common', is_training=is_training, activation='prelu')(top_fc_input)
    # top_input_pre = tf.nn.relu(top_fc_input)

    # bi-inter layer
    fm_input = tf.reshape(top_fc_input, shape=[-1, 3, 300])
    # sum_square part
    summed_features_emb = tf.reduce_sum(fm_input, 1)  # None * K
    summed_features_emb_square = tf.square(summed_features_emb)  # None * K
    # square_sum part
    squared_features_emb = tf.square(fm_input)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K
    # second order
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K

    fc_input = tf.concat([user_top_input, item_top_input, user_item_top_input], axis=1)
    bi_ctr_DNN = DNN(hidden_units=[128], name='bi_inter_ctr', is_training=is_training, activation='prelu')
    bi_ctr = bi_ctr_DNN(y_second_order)
    bi_cvr_DNN = DNN(hidden_units=[128], name='bi_inter_cvr', is_training=is_training, activation='prelu')
    bi_cvr = bi_cvr_DNN(y_second_order)

    # cgc
    gate_vector = tf.concat([item_embeds[2]], axis=1)
    with tf.variable_scope("dnn_common"):
        print("mtl input fc:" + str(fc_input))
        print("mtl input gate:" + str(gate_vector))
        # specific_experts_num, share_experts_num, tasks_name, expert_hiddens, gate_hiddens
        fc_input_mmoe = CGC(2, 2, ["ctr", "cvr"], [256, 128], [12],
                            prefix="CGC0", activation='relu', gate_activation='relu',
                            is_training=is_training)([fc_input,
                                                      gate_vector])

    fc_input_ctr = tf.concat([fc_input_mmoe[0], bi_ctr], axis=1)
    fc_input_cvr = tf.concat([fc_input_mmoe[1], bi_cvr], axis=1)
    print("output click task:" + str(fc_input_ctr))
    print("output order task:" + str(fc_input_cvr))

    with tf.variable_scope('V1'):
        h_ctr = fc_input_ctr
        hidden_layer_dim = [64, 1]
        for i, units in enumerate(hidden_layer_dim):
            layer_name1 = 'top_task_ctr_fc%d' % units
            act = None
            if i < len(hidden_layer_dim) - 1:
                act = tf.nn.relu
            h_ctr = add_fc(h_ctr, units, act, layer_name1)

        h_cvr = fc_input_cvr
        for i, units in enumerate(hidden_layer_dim):
            layer_name2 = 'top_task_cvr_fc%d' % units
            act = None
            if i < len(hidden_layer_dim) - 1:
                act = tf.nn.relu
            h_cvr = add_fc(h_cvr, units, act, layer_name2)

    y_ctr = tf.nn.sigmoid(h_ctr)
    y_cvr = tf.nn.sigmoid(h_cvr)

    loss0 = tf.losses.log_loss(label0, y_ctr, sample_weight0, epsilon=1e-6, reduction=tf.losses.Reduction.SUM)
    loss1 = tf.losses.log_loss(label1, y_cvr, sample_weight1, epsilon=1e-6, reduction=tf.losses.Reduction.SUM)
    '''
    loss2 = tf.losses.mean_squared_error(label2, y_ctr)
    loss3 = tf.losses.mean_squared_error(label3, y_cvr)
    loss00 = student_weight_factor * loss0 + (1 - student_weight_factor) * loss2
    loss11 = student_weight_factor * loss1 + (1 - student_weight_factor) * loss3
    '''
    ##返回部分放入的pred对应的域，其他置为None
    graph_state = kwargs.get('graph_state', None)
    if graph_state == graph_state_helper('PREDICT', 'item_precompute'):
        return [(None, item_input, None, None)]
    if graph_state == graph_state_helper('PREDICT', 'user_precompute'):
        return [(None, user_input, None, None)]
    if graph_state == graph_state_helper('PREDICT', 'user_item_precompute'):
        return [(None, user_item_input, None, None)]
    if graph_state == graph_state_helper('PREDICT', 'evaluate_ctr_cvr'):
        return [(None, y_ctr, label0, sample_weight0), (None, y_cvr, label1, sample_weight1)]
    if graph_state == graph_state_helper('PREDICT', 'predict_top'):
        user_top_input = tf.slice(dense_input, [0, 0], [-1, user_hidden_layer_dim[-1]])
        user_item_top_input = tf.slice(dense_input, [0, user_hidden_layer_dim[-1]],
                                       [-1, user_item_hidden_layer_dim[-1]])
        item_top_input = tf.slice(dense_input, [0, user_hidden_layer_dim[-1] + user_item_hidden_layer_dim[-1]],
                                  [-1, item_hidden_layer_dim[-1]])
        top_fc_input = tf.concat([user_top_input, item_top_input, user_item_top_input], axis=1)
        # top_input_pre = DNN(hidden_units=[300], name='dnn_common', activation='prelu')(top_fc_input)
        # top_input_pre = tf.nn.relu(dense_input)

        # bi-inter layer
        fm_input = tf.reshape(top_fc_input, shape=[-1, 3, 300])
        # sum_square part
        summed_features_emb = tf.reduce_sum(fm_input, 1)  # None * K
        summed_features_emb_square = tf.square(summed_features_emb)  # None * K
        # square_sum part
        squared_features_emb = tf.square(fm_input)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K
        # second order
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K

        fc_input = tf.concat([user_top_input, item_top_input, user_item_top_input], axis=1)
        bi_ctr = bi_ctr_DNN(y_second_order)
        bi_cvr = bi_cvr_DNN(y_second_order)

        # cgc
        gate_vector = tf.concat([item_embeds[2]], axis=1)
        with tf.variable_scope("dnn_common"):
            print("mtl input fc:" + str(fc_input))
            print("mtl input gate:" + str(gate_vector))
            fc_input_mmoe = CGC(2, 2, ["ctr", "cvr"], [256, 128], [12],
                                prefix="CGC0", activation='relu', gate_activation='relu',
                                is_training=is_training)([fc_input,
                                                          gate_vector])

        fc_input_ctr = tf.concat([fc_input_mmoe[0], bi_ctr], axis=1)
        fc_input_cvr = tf.concat([fc_input_mmoe[1], bi_cvr], axis=1)
        print("output click task:" + str(fc_input_ctr))
        print("output order task:" + str(fc_input_cvr))

        with tf.variable_scope('V1'):
            h_ctr = fc_input_ctr
            hidden_layer_dim = [64, 1]
            for i, units in enumerate(hidden_layer_dim):
                layer_name1 = 'top_task_ctr_fc%d' % units
                act = None
                if i < len(hidden_layer_dim) - 1:
                    act = tf.nn.relu
                h_ctr = add_fc(h_ctr, units, act, layer_name1)

            h_cvr = fc_input_cvr
            for i, units in enumerate(hidden_layer_dim):
                layer_name2 = 'top_task_cvr_fc%d' % units
                act = None
                if i < len(hidden_layer_dim) - 1:
                    act = tf.nn.relu
                h_cvr = add_fc(h_cvr, units, act, layer_name2)

        y_ctr = tf.nn.sigmoid(h_ctr)
        y_cvr = tf.nn.sigmoid(h_cvr)
        return [(None, y_ctr, None, None), (None, y_cvr, None, None)]

    else:
        ##(predict,default)状态
        return [(loss0, y_ctr, label0, sample_weight0), (loss1, y_cvr, label1, sample_weight1)]


def main():
    m = build_model()
    tfmlx.save('./output_models/tf_rs_consistency_ts')
    # print(m)


if __name__ == "__main__":
    main()
