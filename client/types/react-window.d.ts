interface FixedSizeListProps {
    height: number | string,
    width: number | string,
    itemCount: number,
    itemSize: number,
    style: React.CSSProperties,
}

interface VariableSizeGridProps {
    columnCount: number,
    columnWidth: (idx: number) => number,
    rowCount: number,
    rowHeight: (idx: number) => number,
    width: number,
    height: number,
}


interface ScrollToGridItem {
    rowIndex: number,
    columnIndex: number,
}

declare module 'react-window' {
    import * as React from "react";

    export class FixedSizeList extends React.Component<FixedSizeListProps> {
        scrollToItem: (a: number) => void
    }

    export class VariableSizeGrid extends React.Component<VariableSizeGridProps> {
        scrollToItem: (indices: ScrollToGridItem) => void
        resetAfterRowIndex(index: number, shouldForceUpdate?: boolean): void
    }
}
