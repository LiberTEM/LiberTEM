import { ReactElement } from "react";
import * as React from "react";

interface PlaceholderProps {
    width: number,
    height: number,
}

const PlaceholderImage: React.SFC<PlaceholderProps> = ({ children, width, height }) => {
    const aspect = 100 * (height / width);
    return (
        <div style={{ paddingBottom: `${aspect}%`, width: "100%", position: "relative" }}>
            {
                React.Children.map(children, child => {
                    if (!React.isValidElement(child)) {
                        return child;
                    }
                    return React.cloneElement(child as ReactElement<any>, {
                        style: {
                            position: "absolute",
                            left: 0,
                            top: 0,
                            ...(child.props as any).style,
                        },
                    });
                })
            }
        </div>
    )
}

export default PlaceholderImage;