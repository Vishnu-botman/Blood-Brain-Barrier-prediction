import React, { useState, useRef, useEffect, useMemo } from 'react';

export default function SearchableSelect({
  options = [],
  selectedIndex = 0,
  onSelect,
  placeholder = "Type to search by drug name, SMILES, or #num...",
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const containerRef = useRef(null);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  const selectedItem = options[selectedIndex] || null;

  // When selectedItem changes externally or closed, sync search query
  useEffect(() => {
    if (!isOpen && selectedItem) {
      setSearchQuery(selectedItem.display_name || selectedItem.name || '');
    }
  }, [selectedItem, isOpen]);

  // Filter options based on query across all 2,050 molecules
  const filteredOptions = useMemo(() => {
    if (!searchQuery.trim() || (!isOpen && selectedItem)) {
      return options.slice(0, 100);
    }
    const q = searchQuery.toLowerCase().trim();
    const matches = [];
    for (let i = 0; i < options.length; i++) {
      const item = options[i];
      const nameMatch = item.name?.toLowerCase().includes(q);
      const smiMatch = item.smiles?.toLowerCase().includes(q);
      const numMatch = String(item.num).includes(q);
      const splitMatch = item.split_tag?.toLowerCase().includes(q);
      if (nameMatch || smiMatch || numMatch || splitMatch) {
        matches.push({ ...item, originalIndex: i });
        if (matches.length >= 120) break;
      }
    }
    return matches;
  }, [options, searchQuery, isOpen, selectedItem]);

  // Handle outside click to close
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
        if (selectedItem) {
          setSearchQuery(selectedItem.display_name || selectedItem.name || '');
        }
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [selectedItem]);

  const handleFocus = () => {
    setIsOpen(true);
    // Select all text for easy replacement
    if (inputRef.current) {
      inputRef.current.select();
    }
  };

  const handleInputChange = (e) => {
    setSearchQuery(e.target.value);
    setIsOpen(true);
    setHighlightedIndex(0);
  };

  const handleSelectOption = (item) => {
    const realIndex = item.originalIndex !== undefined ? item.originalIndex : options.indexOf(item);
    if (realIndex >= 0) {
      onSelect(realIndex, item);
    }
    setSearchQuery(item.display_name || item.name);
    setIsOpen(false);
    if (inputRef.current) {
      inputRef.current.blur();
    }
  };

  const handleKeyDown = (e) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        setIsOpen(true);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.min(prev + 1, filteredOptions.length - 1));
      // Scroll into view
      scrollHighlightedIntoView(highlightedIndex + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.max(prev - 1, 0));
      scrollHighlightedIntoView(highlightedIndex - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filteredOptions[highlightedIndex]) {
        handleSelectOption(filteredOptions[highlightedIndex]);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
      if (selectedItem) {
        setSearchQuery(selectedItem.display_name || selectedItem.name);
      }
    }
  };

  const scrollHighlightedIntoView = (idx) => {
    if (listRef.current) {
      const items = listRef.current.children;
      if (items && items[idx]) {
        items[idx].scrollIntoView({ block: 'nearest' });
      }
    }
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setSearchQuery('');
    setIsOpen(true);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  return (
    <div className="searchable-select-container" ref={containerRef}>
      <div className="searchable-input-wrapper">
        <span className="searchable-icon">🔍</span>
        <input
          ref={inputRef}
          type="text"
          className="searchable-input"
          value={searchQuery}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          autoComplete="off"
          spellCheck="false"
        />
        {searchQuery && (
          <button
            type="button"
            className="searchable-clear-btn"
            onClick={handleClear}
            title="Clear and search"
          >
            ✕
          </button>
        )}
        <span className={`searchable-arrow ${isOpen ? 'open' : ''}`}>▾</span>
      </div>

      {/* Dropdown Options */}
      {isOpen && (
        <div className="searchable-dropdown" ref={listRef}>
          <div className="searchable-dropdown-header">
            <span>
              {searchQuery.trim()
                ? `Matches (${filteredOptions.length} shown)`
                : `Catalog (${options.length.toLocaleString()} molecules)`}
            </span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              Type to filter • ↑↓ to navigate
            </span>
          </div>

          {filteredOptions.length === 0 ? (
            <div className="searchable-empty">
              No compounds found matching "<strong>{searchQuery}</strong>".
              <div style={{ marginTop: '4px', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                Try searching by common name (e.g. <i>diazepam</i>), #num, or switch to Custom SMILES.
              </div>
            </div>
          ) : (
            filteredOptions.map((item, idx) => {
              const isSelected = selectedItem && (item.num === selectedItem.num);
              const isHighlighted = idx === highlightedIndex;

              const splitClass = item.split_tag?.includes('Train')
                ? 'badge-train'
                : item.split_tag?.includes('Test')
                ? 'badge-test'
                : 'badge-valid';

              return (
                <div
                  key={`${item.num}-${idx}`}
                  className={`searchable-option ${isHighlighted ? 'highlighted' : ''} ${isSelected ? 'selected' : ''}`}
                  onClick={() => handleSelectOption(item)}
                  onMouseEnter={() => setHighlightedIndex(idx)}
                >
                  <div className="searchable-option-main">
                    <span className="searchable-option-name">{item.name}</span>
                    <span className="searchable-option-num">#{item.num}</span>
                  </div>

                  <div className="searchable-option-meta">
                    <span className={`badge ${splitClass}`} style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                      {item.split_tag}
                    </span>
                    {item.ground_truth && (
                      <span className="searchable-gt" style={{ 
                        color: item.ground_truth === 'BBB+' ? '#34d399' : '#f87171' 
                      }}>
                        {item.ground_truth}
                      </span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
