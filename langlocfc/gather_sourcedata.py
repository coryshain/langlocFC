import os
import pandas as pd

figs = {
    'fig2': [
        ('PanelA', 'plots/performance_nolangloc_Language_refsim.csv'),
        ('PanelB', 'plots/performance_nolangloc_Language_networksim.csv'),
    ],
    'fig3': [
        ('PanelA_topleft', 'plots/performance_nolangloc_Language_sim.csv'),
        ('PanelA_bottomleft', 'plots/performance_nolangloc_Language_contrast.csv'),
        ('PanelA_topright', 'plots/performance_nonlinguistic_Language_sim.csv'),
        ('PanelA_bottomright', 'plots/performance_nonlinguistic_Language_contrast.csv'),
        ('PanelB_left', 'plots/performance_nolangloc_Auditory_contrast.csv'),
        ('PanelB_middle', 'plots/performance_nolangloc_MD_contrast.csv'),
        ('PanelB_right', 'plots/performance_nolangloc_ToM_contrast.csv'),
        ('PanelC_top', 'plots/performance_nonlinguistic_LANA_nvoxels_by_n_networks_grid.csv'),
        ('PanelC_bottom', 'plots/performance_nonlinguistic_LANA_eval_Lang_S-N_by_n_networks_sim_grid.csv'),
    ],
    'fig4': [
        ('PanelA', 'plots/stability_within_between.csv'),
        ('PanelB', 'plots/runwise_correlations.csv'),
        ('PanelC_1run_zr', 'plots/performance_nolangloc_multisession01_Language_sim.csv'),
        ('PanelC_2run_zr', 'plots/performance_nolangloc_multisession02_Language_sim.csv'),
        ('PanelC_5run_zr', 'plots/performance_nolangloc_multisession05_Language_sim.csv'),
        ('PanelC_10run_zr', 'plots/performance_nolangloc_multisession10_Language_sim.csv'),
        ('PanelC_25run_zr', 'plots/performance_nolangloc_multisession25_Language_sim.csv'),
        ('PanelC_50run_zr', 'plots/performance_nolangloc_multisession50_Language_sim.csv'),
        ('PanelC_1run_t', 'plots/performance_nolangloc_multisession01_Language_contrast.csv'),
        ('PanelC_2run_t', 'plots/performance_nolangloc_multisession02_Language_contrast.csv'),
        ('PanelC_5run_t', 'plots/performance_nolangloc_multisession05_Language_contrast.csv'),
        ('PanelC_10run_t', 'plots/performance_nolangloc_multisession10_Language_contrast.csv'),
        ('PanelC_25run_t', 'plots/performance_nolangloc_multisession25_Language_contrast.csv'),
        ('PanelC_50run_t', 'plots/performance_nolangloc_multisession50_Language_contrast.csv'),
    ],
    'fig5': [
        ('PanelB_IFGorb', 'plots/pdd/PDD_nlength2_fROI_IFGorb_plot.csv'),
        ('PanelB_IFGtri', 'plots/pdd/PDD_nlength2_fROI_IFGtri_plot.csv'),
        ('PanelB_TP', 'plots/pdd/PDD_nlength2_fROI_TP_plot.csv'),
        ('PanelB_aSTS', 'plots/pdd/PDD_nlength2_fROI_aSTS_plot.csv'),
        ('PanelB_pSTS', 'plots/pdd/PDD_nlength2_fROI_pSTS_plot.csv'),
        ('PanelB_TPJ', 'plots/pdd/PDD_nlength2_fROI_TPJ_plot.csv'),
        ('PanelC_IFGorb', 'plots/pdd/PDD_nlength2_IFGorb_plot.csv'),
        ('PanelC_IFGtri', 'plots/pdd/PDD_nlength2_IFGtri_plot.csv'),
        ('PanelC_TP', 'plots/pdd/PDD_nlength2_TP_plot.csv'),
        ('PanelC_aSTS', 'plots/pdd/PDD_nlength2_aSTS_plot.csv'),
        ('PanelC_pSTS', 'plots/pdd/PDD_nlength2_pSTS_plot.csv'),
        ('PanelC_TPJ', 'plots/pdd/PDD_nlength2_TPJ_plot.csv'),
        ('PanelD_IFGorb', 'plots/pdd/PDD_nlength2_ROI_IFGorb_plot.csv'),
        ('PanelD_IFGtri', 'plots/pdd/PDD_nlength2_ROI_IFGtri_plot.csv'),
        ('PanelD_TP', 'plots/pdd/PDD_nlength2_ROI_TP_plot.csv'),
        ('PanelD_aSTS', 'plots/pdd/PDD_nlength2_ROI_aSTS_plot.csv'),
        ('PanelD_pSTS', 'plots/pdd/PDD_nlength2_ROI_pSTS_plot.csv'),
        ('PanelD_TPJ', 'plots/pdd/PDD_nlength2_ROI_TPJ_plot.csv'),
    ],
    'figS2': [
        ('LANG_zr', 'plots/performance_oracle_LANG_sim.csv'),
        ('LANA_zr', 'plots/performance_oracle_LANA_sim.csv'),
        ('LANA_zr', 'plots/performance_oracle_Lang_S-N_sim.csv'),
        ('LANG_t', 'plots/performance_oracle_LANG_contrast.csv'),
        ('LANA_t', 'plots/performance_oracle_LANA_contrast.csv'),
        ('LANA_t', 'plots/performance_oracle_Lang_S-N_contrast.csv'),
    ],
    'figS3': [
        ('Unresidualized_zr', 'plots/performance_unresidualized_Language_sim.csv'),
        ('Unresidualized_t', 'plots/performance_unresidualized_Language_contrast.csv'),
        ('Residualized_zr', 'plots/performance_residualized_Language_sim.csv'),
        ('Residualized_t', 'plots/performance_residualized_Language_contrast.csv'),
    ],
    'figS4': [
        ('RawTimecourse_zr', 'plots/performance_nolanglocSearchNobpTimecourse_Language_sim.csv'),
        ('BandpassedTimecourse_zr', 'plots/performance_nolanglocSearchBpTimecourse_Language_sim.csv'),
        ('Downsampled_zr', 'plots/performance_nolanglocSearchConnDownsample_Language_sim.csv'),
        ('DownsampledBinarized_zr', 'plots/performance_nolanglocSearchConnDownsampleBin_Language_sim.csv'),
        ('Parcels_zr', 'plots/performance_nolanglocSearchConnRegions_Language_sim.csv'),
        ('ParcelsBinarized_zr', 'plots/performance_nolanglocSearchConnRegionsBin_Language_sim.csv'),
        ('RawTimecourse_t', 'plots/performance_nolanglocSearchNobpTimecourse_Language_contrast.csv'),
        ('BandpassedTimecourse_t', 'plots/performance_nolanglocSearchBpTimecourse_Language_contrast.csv'),
        ('Downsampled_t', 'plots/performance_nolanglocSearchConnDownsample_Language_contrast.csv'),
        ('DownsampledBinarized_t', 'plots/performance_nolanglocSearchConnDownsampleBin_Language_contrast.csv'),
        ('Parcels_t', 'plots/performance_nolanglocSearchConnRegions_Language_contrast.csv'),
        ('ParcelsBinarized_t', 'plots/performance_nolanglocSearchConnRegionsBin_Language_contrast.csv'),
    ],
    'figS5': [
        ('OurApproach_zr', 'plots/performance_templateMatchingBaseline_LanA_sim.csv'),
        ('OurApproach_t', 'plots/performance_templateMatchingBaseline_LanA_contrast.csv'),
        ('TemplateMatching_zr', 'plots/performance_templateMatchingMain_LanA_sim.csv'),
        ('TemplateMatching_t', 'plots/performance_templateMatchingMain_LanA_contrast.csv'),
    ],
    'figS6': [
        ('LanA_Rest_zr', 'plots/performance_rest_v_task_rest_LanA_sim.csv'),
        ('AUD_Rest_zr', 'plots/performance_rest_v_task_rest_AUD_sim.csv'),
        ('FPNA_Rest_zr', 'plots/performance_rest_v_task_rest_FPN-A_sim.csv'),
        ('DNB_Rest_zr', 'plots/performance_rest_v_task_rest_DN-B_sim.csv'),
        ('LanA_Rest_t', 'plots/performance_rest_v_task_rest_LanA_contrast.csv'),
        ('AUD_Rest_t', 'plots/performance_rest_v_task_rest_AUD_contrast.csv'),
        ('FPNA_Rest_t', 'plots/performance_rest_v_task_rest_FPN-A_contrast.csv'),
        ('DNB_Rest_t', 'plots/performance_rest_v_task_rest_DN-B_contrast.csv'),
        ('LanA_Task_zr', 'plots/performance_rest_v_task_task_LanA_sim.csv'),
        ('AUD_Task_zr', 'plots/performance_rest_v_task_task_AUD_sim.csv'),
        ('FPNA_Task_zr', 'plots/performance_rest_v_task_task_FPN-A_sim.csv'),
        ('DNB_Task_zr', 'plots/performance_rest_v_task_task_DN-B_sim.csv'),
        ('LanA_Task_t', 'plots/performance_rest_v_task_task_LanA_contrast.csv'),
        ('AUD_Task_t', 'plots/performance_rest_v_task_task_AUD_contrast.csv'),
        ('FPNA_Task_t', 'plots/performance_rest_v_task_task_FPN-A_contrast.csv'),
        ('DNB_Task_t', 'plots/performance_rest_v_task_task_DN-B_contrast.csv'),
    ],
}

if not os.path.exists('sourcedata'):
    os.makedirs('sourcedata')

for fig in figs:
    with pd.ExcelWriter(f'sourcedata/{fig}.xlsx', engine='openpyxl') as writer:
        for sheet_title, source_data in figs[fig]:
            df = pd.read_csv(source_data)
            df.to_excel(writer, sheet_name=sheet_title, index=False)

