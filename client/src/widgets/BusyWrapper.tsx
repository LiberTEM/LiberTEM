import * as React from "react";
import BusySpinner from "./BusySpinner";

interface BusyWrapperProps {
    busy: boolean,
    children?: React.ReactNode,
}

const BusyWrapper: React.FC<BusyWrapperProps> = ({ children, busy }) => {
    const styles: React.CSSProperties = {
        position: "relative",
    };
    return (
        <div style={styles}>
            {children}
            {busy && <BusySpinner />}
        </div>
    )
}

export default BusyWrapper;