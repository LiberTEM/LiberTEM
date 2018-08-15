import * as React from "react";

export interface FixedSizeListProps {
    height: number | string,
    width: number | string,
    itemCount: number,
    itemSize: number,
}

declare module 'react-window' {
    export class FixedSizeList extends React.Component<FixedSizeListProps> {
        scrollToItem: (a: number) => void
    }
}
