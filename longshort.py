from pytz import timezone
from zipline.utils.tradingcalendar import get_early_closes
from quantopian.pipeline import Pipeline, classifiers
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.morningstar import operation_ratios, valuation, balance_sheet, income_statement, valuation_ratios, company_reference
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data import morningstar
from quantopian.pipeline import CustomFactor 
import numpy as np
import pandas as pd
import quantopian.experimental.optimize as opt


import statsmodels.api as sm
class Sector(CustomFactor):
    inputs = [ morningstar.asset_classification.morningstar_sector_code ]
    window_length = 1
    def compute(self, today, assets, out, sector):
        out[:] = sector

        
class Momentum(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252
    def compute(self, today, assets, out, close):       
        out[:] = close[-20] / close[0]
        
def make_pipeline():
    sectors = Sector()
    last_p = USEquityPricing.close.latest
    peg = valuation_ratios.peg_payback.latest.zscore(groupby = classifiers.morningstar.Sector())
    shortable = (last_p > 5)
    PriceTo52week = Momentum().zscore(groupby = classifiers.morningstar.Sector())
    PEs = valuation_ratios.pe_ratio.latest.zscore(groupby = classifiers.morningstar.Sector())
    s = valuation.market_cap.latest
    PBs = valuation_ratios.pb_ratio.latest.zscore(groupby = classifiers.morningstar.Sector())
    Ebitda_ratio = valuation_ratios.ev_to_ebitda.latest.zscore(groupby = classifiers.morningstar.Sector())
    ROA = operation_ratios.roa.latest.zscore(groupby = classifiers.morningstar.Sector())
    symbols = company_reference.primary_symbol.latest
    screen_size = (s > 200)
    combo_rank = (peg) + (Ebitda_ratio) - (PriceTo52week)
    return Pipeline(
    columns={
        'combo_rank' : combo_rank,
            'sector': sectors,
            'symbols' : symbols
    },
        screen= (shortable & 
                 screen_size
                )
    )

def initialize(context):
    context.counter = 0
    pipe = make_pipeline()
    attach_pipeline(pipe, 'first_pipe')
    schedule_function(func=rebalance,
                      date_rule=date_rules.month_start(), 
                      time_rule=time_rules.market_open(hours=1,minutes=0), 
                      half_days=True)
    # set my leverage
    context.long_leverage = 1.0
    context.short_leverage = -1.0
    
    # List of sectors
    context.sector_mappings = [ 
         101.0,  
         102.0,  
         103.0,  
         104.0,  
         205.0,  
         206.0,  
         207.0,  
         308.0,  
         309.0,  
         310.0,  
         311.0  
    ]

def before_trading_start(context, data):
    # Call pipelive_output to get the output
    context.output = pipeline_output('first_pipe')
    #Lower scoring securites are bought, higher scoring are shorted
    context.long_list = context.output.sort(['combo_rank'], ascending=True).iloc[:150]
    context.short_list = context.output.sort(['combo_rank'], ascending=True).iloc[-150:]   
    
    update_universe(context.long_list.index.union(context.short_list.index))
    
    
# This rebalancing is called according to our schedule_function settings.     
def rebalance(context,data):
    log.info(context.account.leverage)
    context.counter = (context.counter + 1) % 11
    short_weight = context.short_leverage / float(len(context.short_list)) 
    long_weight = context.long_leverage / float(len(context.long_list))
    targets = {}
    #Add longs to targets
    for long_stock in context.long_list.index:
        if long_stock in data:
            targets[long_stock] = long_weight
    #Add shorts to targets
    for short_stock in context.short_list.index:
        if short_stock in data:
            targets[short_stock] = short_weight

    order_optimal_portfolio(objective=opt.TargetPortfolioWeights(targets),
                           constraints=[])
