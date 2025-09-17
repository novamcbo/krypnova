"""
Exchange Connector Module for Krypnova

This module provides the ExchangeConnector class that manages connections
to multiple cryptocurrency and financial exchanges including Binance, Kraken,
Coinbase, and Alpaca.
"""

import logging
import asyncio
from typing import Dict, Any, Optional


def detect_exchange(symbol: str) -> str:
    """
    Detect the appropriate exchange based on the symbol format.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSD', 'BTC-USD', 'XBTUSD')
    
    Returns:
        str: Exchange name ('binance', 'kraken', 'coinbase', or 'alpaca')
    """
    symbol_upper = symbol.upper()
    
    # Kraken specific symbols
    if symbol_upper in ['XBTUSD', 'XETHZ', 'XXBTZUSD', 'XETHZUSD']:
        return 'kraken'
    
    # Coinbase format (with dash)
    if '-' in symbol:
        return 'coinbase'
    
    # Stock symbols (for Alpaca)
    if symbol_upper in ['SPY', 'NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL']:
        return 'alpaca'
    
    # Default to binance for crypto without dashes
    return 'binance'


class ExchangeConnector:
    """
    Main connector class for handling multiple exchange connections.
    
    This class manages individual exchange clients and provides unified
    access to trading data across multiple platforms.
    """
    
    def __init__(self):
        """Initialize the exchange connector with individual exchange services."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual exchange services
        # These would be actual exchange client instances in a real implementation
        self.binance = self._create_mock_binance_client()
        self.kraken = self._create_mock_kraken_client()
        self.coinbase = self._create_mock_coinbase_client()
        self.alpaca = self._create_mock_alpaca_client()
    
    def _create_mock_binance_client(self):
        """Create a mock Binance client for testing."""
        class MockBinanceClient:
            async def get_trade_fee(self):
                return [{"makerCommission": "0.001", "takerCommission": "0.001"}]
        return MockBinanceClient()
    
    def _create_mock_kraken_client(self):
        """Create a mock Kraken client for testing."""
        class MockKrakenClient:
            async def query_private(self, method):
                if method == "TradeVolume":
                    return {
                        "fees": {
                            "XXBTZUSD": {
                                "fee_maker": "0.16",
                                "fee": "0.26"
                            }
                        }
                    }
        return MockKrakenClient()
    
    def _create_mock_coinbase_client(self):
        """Create a mock Coinbase client for testing."""
        class MockCoinbaseClient:
            async def get_fees(self):
                # Coinbase Pro standard fees
                return {"maker": 0.004, "taker": 0.006}
        return MockCoinbaseClient()
    
    def _create_mock_alpaca_client(self):
        """Create a mock Alpaca client for testing."""
        class MockAlpacaClient:
            async def get_fees(self):
                # Alpaca typically has no fees for crypto/stock trading
                return {"maker": 0.0, "taker": 0.0}
        return MockAlpacaClient()

    async def get_fees(self, exchange: str) -> dict:
        """
        Get trading fees for a specific exchange.
        
        Args:
            exchange: Exchange name ('binance', 'kraken', 'coinbase', 'alpaca')
        
        Returns:
            dict: Dictionary with 'maker' and 'taker' fee rates
        """
        try:
            if exchange == "binance":
                info = await self.binance.get_trade_fee()
                return {"maker": float(info[0]["makerCommission"]), "taker": float(info[0]["takerCommission"])}
            elif exchange == "kraken":
                info = await self.kraken.query_private("TradeVolume")
                return {"maker": float(info["fees"]["XXBTZUSD"]["fee_maker"]) / 100,
                        "taker": float(info["fees"]["XXBTZUSD"]["fee"]) / 100}
            elif exchange == "coinbase":
                info = await self.coinbase.get_fees()
                return {"maker": info["maker"], "taker": info["taker"]}
            elif exchange == "alpaca":
                info = await self.alpaca.get_fees()
                return {"maker": info["maker"], "taker": info["taker"]}
            else:
                self.logger.warning(f"[ExchangeConnector] ❌ Exchange no soportado: {exchange}")
                return {"maker": 0.001, "taker": 0.001}
        except Exception as e:
            self.logger.warning(f"[ExchangeConnector] ❌ No se pudieron obtener fees dinámicos para {exchange}: {e}")
            return {"maker": 0.001, "taker": 0.001}

    async def get_ohlcv_by_exchange(self, exchange: str, symbol: str) -> Dict:
        """
        Get OHLCV data for a symbol from a specific exchange.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
        
        Returns:
            Dict: OHLCV data
        """
        # Mock implementation for now
        self.logger.info(f"[ExchangeConnector] Getting OHLCV for {symbol} from {exchange}")
        return {}
    
    async def get_symbols_for_exchange(self, exchange: str) -> list:
        """
        Get available symbols for a specific exchange.
        
        Args:
            exchange: Exchange name
        
        Returns:
            list: Available trading symbols
        """
        # Mock implementation
        symbols_map = {
            'binance': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            'kraken': ['XBTUSD', 'ETHUSD', 'ADAUSD'],
            'coinbase': ['BTC-USD', 'ETH-USD', 'ADA-USD'],
            'alpaca': ['SPY', 'NVDA', 'AAPL']
        }
        return symbols_map.get(exchange, [])
    
    async def get_orderbook(self, exchange: str, symbol: str) -> dict:
        """
        Get orderbook data for a symbol from a specific exchange.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
        
        Returns:
            dict: Orderbook data
        """
        self.logger.info(f"[ExchangeConnector] Getting orderbook for {symbol} from {exchange}")
        return {"bids": [], "asks": []}
    
    async def resolve_symbol(self, exchange: str, symbol: str) -> str:
        """
        Resolve symbol format for a specific exchange.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
        
        Returns:
            str: Resolved symbol
        """
        # Simple symbol resolution logic
        if exchange == "kraken" and symbol.upper() in ["BTCUSD", "BTC-USD"]:
            return "XBTUSD"
        elif exchange == "coinbase" and not "-" in symbol:
            # Convert BTCUSD to BTC-USD format for Coinbase
            if len(symbol) >= 6:
                return f"{symbol[:3]}-{symbol[3:]}"
        return symbol
    
    async def get_orderbook_df(self, exchange: str, symbol: str) -> Dict:
        """
        Get orderbook data as DataFrame.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
        
        Returns:
            Dict: Orderbook data
        """
        self.logger.info(f"[ExchangeConnector] Getting orderbook DataFrame for {symbol} from {exchange}")
        return {}
    
    async def get_all_symbols(self) -> dict:
        """
        Get all available symbols from all exchanges.
        
        Returns:
            dict: Dictionary mapping exchange names to symbol lists
        """
        all_symbols = {}
        for exchange in ['binance', 'kraken', 'coinbase', 'alpaca']:
            try:
                all_symbols[exchange] = await self.get_symbols_for_exchange(exchange)
            except Exception as e:
                self.logger.warning(f"Error getting symbols for {exchange}: {e}")
                all_symbols[exchange] = []
        return all_symbols